import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import FigManager

class SAC(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,

            pixel_shape=None,
            use_discrete_sac: bool = False,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size

        self._reward_scale_factor = scale_reward

        self.use_discrete_sac = use_discrete_sac
        if self.use_discrete_sac:
            self._target_entropy = np.log(self._env_spec.action_space.n) * target_coef
        else:
            self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_train_trajectories_kwargs(self, runner):
        extras = [{} for _ in range(runner._train_args.batch_size)]

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_op(tensors, v)

        return tensors

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
            )

        # LossSacp should be updated here because Q functions are changed by optimizers.
        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self.option_policy.process_observations(v['obs'])
        next_processed_cat_obs = self.option_policy.process_observations(v['next_obs'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
            use_discrete_sac=self.use_discrete_sac
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self.option_policy.process_observations(v['obs'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
            use_discrete_sac=self.use_discrete_sac,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v, use_discrete_sac=self.use_discrete_sac,
        )

    def _evaluate_policy(self, runner, **kwargs):
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=[{} for _ in range(self.num_random_trajectories)],
            worker_update=dict(
                _render=False,
                _deterministic_initial_state=False,
                _deterministic_policy=True, #NOTE: changed from original: self.eval_deterministic_traj,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, np.zeros((self.num_random_trajectories, 3)), self.eval_plot_axis, fm.ax
            )

        eval_option_metrics = {}
        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
