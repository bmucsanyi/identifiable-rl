import torch
from torch.nn import functional as F

import global_context
from garage import TrajectoryBatch
from garage.torch import compute_advantages, filter_valids
from garagei import log_performance_ex
from iod.iod import IOD
import numpy as np

from iod.utils import FigManager


class PPO(IOD):
    def __init__(
            self,
            *,
            vf,
            gae_lambda,
            ppo_clip,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.vf = vf.to(self.device)
        self.param_modules.update(vf=self.vf)
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.replay_buffer = None

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

    def _train_once_inner(self, path_data):
        # Update advantages
        valids = [len(traj) for traj in path_data['obs']]
        obs_flat = torch.tensor(np.concatenate(path_data['obs'], axis=0), dtype=torch.float32, device=self.device)
        v_input = self.option_policy.process_observations(obs_flat)
        v_pred = self.vf(v_input).flatten().cpu()
        v_pred = v_pred.split(valids, dim=0)
        v_pred = torch.nn.utils.rnn.pad_sequence(v_pred, batch_first=True, padding_value=0.0)

        rewards = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(rewards, dtype=torch.float32) for rewards in path_data['rewards']],
            batch_first=True,
            padding_value=0.0
        )

        advantages = compute_advantages(self.discount, self.gae_lambda, self.max_path_length, v_pred, rewards)

        path_data['advantages'] = list(advantages.detach().cpu().numpy())

        epoch_data = self._flatten_data(path_data)

        for _ in range(self._trans_optimization_epochs):
            tensors = {}
            v = self._get_mini_tensors(epoch_data)

            loss_op_key = self._update_loss_op(tensors, v)
            self._gradient_descent(
                tensors[loss_op_key],
                optimizer_keys=['option_policy'],
            )

            loss_vf_key = self._update_loss_vf(tensors, v)
            self._gradient_descent(
                tensors[loss_vf_key],
                optimizer_keys=['vf'],
            )

        return tensors

    def _update_loss_vf(self, tensors, v):
        v_input = self.option_policy.process_observations(v['obs'])
        v_pred = self.vf(v_input).flatten()

        v_target = v['returns']

        loss_vf = F.mse_loss(v_pred, v_target)

        tensors.update({
            'LossVf': loss_vf,
        })
        return 'LossVf'

    def _update_loss_op(self, tensors, v):
        old_ll = v['log_probs']
        new_ll = self.option_policy(v['obs'])[0].log_prob(v['actions'])

        likelihood_ratio = (new_ll - old_ll).exp()

        advantages = v['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio, min=1 - self.ppo_clip, max=1 + self.ppo_clip)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        loss_op = -torch.min(surrogate, surrogate_clip)

        policy_entropies = self.option_policy(v['obs'])[0].entropy()
        loss_op = loss_op - self.alpha * policy_entropies

        loss_op = loss_op.mean()

        loss_key = 'LossSurrogateOp'
        tensors.update({
            loss_key: loss_op,
        })
        return loss_key

    def _evaluate_policy(self, runner, **kwargs):
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=[{} for _ in range(self.num_random_trajectories)],
            worker_update=dict(
                _render=False,
                _deterministic_initial_state=False,
                _deterministic_policy=True,
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
