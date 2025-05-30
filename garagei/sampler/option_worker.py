import functools
import numpy as np
from garage.experiment import deterministic

import torch
from garage.sampler import DefaultWorker

from iod.utils import get_np_concat_obs


class OptionWorker(DefaultWorker):
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number,
            sampler_key,
    ):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._sampler_key = sampler_key
        self._max_path_length_override = None
        self._cur_extras = None
        self._cur_extra_idx = None
        self._cur_extra_keys = set()
        self._render = False
        self._deterministic_policy = None
        self._ground_truth_states = []
        self._encoder_outputs = []

    def update_env(self, env_update):
        if env_update is not None:
            if isinstance(env_update, dict):
                for k, v in env_update.items():
                    setattr(self.env, k, v)
            else:
                super().update_env(env_update)

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            deterministic.set_seed(self._seed + self._worker_number * 10000)

    def update_worker(self, worker_update):
        if worker_update is not None:
            if isinstance(worker_update, dict):
                for k, v in worker_update.items():
                    setattr(self, k, v)
                    if k == '_cur_extras':
                        if v is None:
                            self._cur_extra_keys = set()
                        else:
                            if len(self._cur_extras) > 0:
                                self._cur_extra_keys = set(self._cur_extras[0].keys())
                            else:
                                self._cur_extra_keys = None

            else:
                raise TypeError('Unknown worker update type.')

    def get_attrs(self, keys):
        attr_dict = {}
        for key in keys:
            attr_dict[key] = functools.reduce(getattr, [self] + key.split('.'))
        return attr_dict

    def start_rollout(self):
        """Begin a new rollout."""
        if 'goal' in self._cur_extra_keys:
            goal = self._cur_extras[self._cur_extra_idx]['goal']
            reset_kwargs = {'goal': goal}
        else:
            reset_kwargs = {}

        env = self.env
        while hasattr(env, 'env'):
            env = getattr(env, 'env')

        self._path_length = 0
        self._prev_obs = self.env.reset(**reset_kwargs)
        self._prev_extra = None

        self.goal_rep = None
        if self._cur_extras is not None and 'traj_encoder' in self._cur_extras[0]:
            traj_encoder = self._cur_extras[0]['traj_encoder'].to('cpu')
            obs = self._prev_obs[..., :self._prev_obs.shape[-1] // 2]
            self.obs_dim = obs.shape[-1]
            self.goal = self._prev_obs[..., self._prev_obs.shape[-1] // 2:]
            with torch.no_grad():
                self.goal_rep = traj_encoder(torch.from_numpy(self.goal)).mean.numpy()
            self._prev_obs = np.concatenate([obs, self.goal_rep], axis=-1)
            self._cur_extra_keys = set()

        self.agent.reset()

        # Clear the lists for a new rollout
        self._ground_truth_states = []
        self._encoder_outputs = []

        # Store initial state
        obs = self._prev_obs
        if hasattr(self, 'obs_dim'):
            obs = self._prev_obs[..., :self.obs_dim]

        if hasattr(self.env, "ground_truth_state") and self.env.ground_truth_state is not None:
            self._ground_truth_states.append(self.env.ground_truth_state)
        else:
            self._ground_truth_states.append(obs)

        # Get encoder output if available
        if self.encoder is not None:
            with torch.no_grad():
                initial_encoder_output = self.encoder(torch.from_numpy(obs)).mean.numpy()
            self._encoder_outputs.append(initial_encoder_output)

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_path_length`.

        """
        cur_max_path_length = self._max_path_length if self._max_path_length_override is None else self._max_path_length_override
        if self._path_length < cur_max_path_length:
            if 'option' in self._cur_extra_keys:
                cur_extra_key = 'option'
            else:
                cur_extra_key = None

            if cur_extra_key is None:
                agent_input = self._prev_obs
            else:
                if isinstance(self._cur_extras[self._cur_extra_idx][cur_extra_key], list):
                    cur_extra = self._cur_extras[self._cur_extra_idx][cur_extra_key][self._path_length]
                    if cur_extra is None:
                        cur_extra = self._prev_extra
                        self._cur_extras[self._cur_extra_idx][cur_extra_key][self._path_length] = cur_extra
                else:
                    cur_extra = self._cur_extras[self._cur_extra_idx][cur_extra_key]

                agent_input = get_np_concat_obs(
                    self._prev_obs, cur_extra,
                )
                self._prev_extra = cur_extra

            if self._deterministic_policy is not None:
                self.agent._force_use_mode_actions = self._deterministic_policy

            a, agent_info = self.agent.get_action(agent_input)

            if self._render:
                next_o, r, d, env_info = self.env.step(a, render=self._render)
            else:
                next_o, r, d, env_info = self.env.step(a)

            # Record ground-truth state and encoder output
            if self.goal_rep is not None:
                # Extract state without goal
                state = next_o[..., :next_o.shape[-1] // 2]

                if hasattr(self.env, "ground_truth_state") and self.env.ground_truth_state is not None:
                    self._ground_truth_states.append(self.env.ground_truth_state)
                else:
                    self._ground_truth_states.append(state)

                # Get encoder representation if available
                if self.encoder is not None:
                    with torch.no_grad():
                        encoder_output = self.encoder(torch.from_numpy(state)).mean.numpy()
                    self._encoder_outputs.append(encoder_output)
            else:
                # Use full observation as state
                if hasattr(self.env, "ground_truth_state") and self.env.ground_truth_state is not None:
                    self._ground_truth_states.append(self.env.ground_truth_state)
                else:
                    self._ground_truth_states.append(next_o)

                # Get encoder representation if available
                if self.encoder is not None:
                    with torch.no_grad():
                        encoder_output = self.encoder(torch.from_numpy(next_o)).mean.numpy()
                    self._encoder_outputs.append(encoder_output)

            if self.goal_rep is not None:
                self._observations.append(np.concatenate([self._prev_obs[..., :self.obs_dim], self.goal], axis=-1))
            else:
                self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)

            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k in self._cur_extra_keys:
                if isinstance(self._cur_extras[self._cur_extra_idx][k], list):
                    self._agent_infos[k].append(self._cur_extras[self._cur_extra_idx][k][self._path_length])
                else:
                    self._agent_infos[k].append(self._cur_extras[self._cur_extra_idx][k])

            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            if not d:
                if self.goal_rep is not None:
                    obs = next_o[..., :next_o.shape[-1] // 2]
                    next_o = np.concatenate([obs, self.goal_rep], axis=-1)
                self._prev_obs = next_o
                return False
        self._terminals[-1] = True
        self._lengths.append(self._path_length)
        if self.goal_rep is not None:
            self._last_observations.append(np.concatenate([self._prev_obs[..., :self.obs_dim], self.goal], axis=-1))
        else:
            self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        if self._cur_extras is not None:
            self._cur_extra_idx += 1
        self.start_rollout()
        while not self.step_rollout():
            pass
        return self.collect_rollout()
