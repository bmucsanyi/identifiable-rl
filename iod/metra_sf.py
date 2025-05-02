from typing import Dict

import torch
from torch.nn import functional as F

from iod.metra import METRA
from iod.sac_utils import _clip_actions

class MetraSf(METRA):
    """This class implements a version of METRA that uses successor features to learn a policy instead of relying on SAC.
    """
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def _train_components(self, epoch_data: Dict[str, torch.tensor]) -> Dict:
        """If replay buffer is full enough, train all components sequentially for _trans_optimization_epochs number of times.

        Args:
            epoch_data (Dict[str, torch.tensor]): Dict containing the data for the current epoch.

        Returns:
            Dict: Dict containing the training losses etc. for each component.
        """
        # Make sure replay buffer is used and has enough transitions
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            train_store = {}

            # Sample mini batch
            if self.replay_buffer is None:
                mini_batch = self._get_mini_tensors(epoch_data)
            else:
                mini_batch = self._sample_replay_buffer()

            # Update trajectory encoder
            self._optimize_te(train_store, mini_batch)

            # Update successor features
            self._optimize_sf(train_store, mini_batch)

            # Optimize the policy
            self._optimize_op(train_store, mini_batch)

        return train_store

    def _optimize_sf(self, train_store: Dict, mini_batch: Dict) -> None:
        """Computes the successor feature loss and optimizes it with gradient descent.

        Args:
            train_store (Dict): train store
            mini_batch (Dict): mini batch data
        """
        # Compute successor feature loss
        self._update_loss_sf_td(train_store, mini_batch)

        # Take gradient step on successor feature networks
        self._gradient_descent(
            train_store['LossQf1'] + train_store['LossQf2'],
            optimizer_keys=['qf'],
        )

        # Update target networks
        self._update_targets()

    def _optimize_op(self, train_store: Dict, mini_batch: Dict) -> None:
        """Optimizes the policy and the entropy coefficient.

        Args:
            train_store (Dict): train store
            mini_batch (Dict): mini batch data
        """
        # Concatenate observations and options
        states = self._get_concat_obs(self.option_policy.process_observations(mini_batch['obs']), mini_batch['options'])

        # Sample actions from the policy
        action_dists, *_ = self.option_policy(states)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = _clip_actions(self, new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)

        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        # Compute successor features
        sf1_feature = self.qf1(states, new_actions)
        sf2_feature = self.qf2(states, new_actions)

        # Get Q values from successor features
        q1_values = (sf1_feature * mini_batch['options']).sum(dim=-1)
        q2_values = (sf2_feature * mini_batch['options']).sum(dim=-1)
        q_values = torch.min(q1_values, q2_values)

        # Compute policy loss
        logits = -1 * q_values + alpha * new_action_log_probs

        loss_op = logits.mean()

        train_store.update({
            'LossOp': loss_op,
        })

        mini_batch.update({
            'new_action_log_probs': new_action_log_probs
        })

        self._gradient_descent(
            loss_op,
            optimizer_keys=['option_policy'],
        )

        # Automatically tune entropy coefficient
        self._update_loss_alpha(train_store, mini_batch)
        self._gradient_descent(
            train_store['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

    def _update_loss_sf_td(self, train_store: Dict, mini_batch: Dict) -> None:
        """Computes the successor feature loss.

        Args:
            train_store (Dict): train store
            mini_batch (Dict): mini batch data
        """
        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']
        actions = mini_batch['actions']
        options = mini_batch['options']
        next_options = mini_batch['next_options']
        dones = mini_batch['dones']
        assert torch.allclose(options, next_options)

        # Concatenate observations and options
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(obs), options)
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(next_obs),
                                                      next_options)

        # Compute observation representations
        cur_repr = self.traj_encoder(obs).mean
        next_repr = self.traj_encoder(next_obs).mean

        # Compute successor features
        sf1_pred = self.qf1(processed_cat_obs, actions)
        sf2_pred = self.qf2(processed_cat_obs, actions)

        # Sample actions from the policy
        next_action_dists, *_ = self.option_policy(next_processed_cat_obs)
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            _, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = _clip_actions(self, new_next_actions)

        # Compute target successor features
        target_next_sf1 = self.target_qf1(next_processed_cat_obs, new_next_actions)
        target_next_sf2 = self.target_qf2(next_processed_cat_obs, new_next_actions)
        target_next_sf = torch.stack([target_next_sf1, target_next_sf2], dim=1)
        next_qf1_values = (target_next_sf1 * next_options).sum(dim=-1)
        next_qf2_values = (target_next_sf2 * next_options).sum(dim=-1)
        next_qf_values = torch.stack([next_qf1_values, next_qf2_values], dim=1)
        min_idxs = torch.argmin(next_qf_values, dim=1)
        target_next_sf_min = target_next_sf[torch.arange(self._trans_minibatch_size, device=self.device), min_idxs]
        target_next_sf = target_next_sf_min

        with torch.no_grad():
            if self.turn_off_dones:
                dones[...] = 0

            # Compute successor feature targets
            if self.metra_mlp_rep:
                sf_target = self.f_encoder(mini_batch['obs'], mini_batch['next_obs']) + self.discount * (1. - dones[:, None]) * target_next_sf
            elif self.no_diff_in_rep:
                sf_target = cur_repr + self.discount * (1. - dones[:, None]) * target_next_sf
            else:
                sf_target = (next_repr - cur_repr) + self.discount * (1. - dones[:, None]) * target_next_sf

        # Compute successor feature losses
        loss_sf1 = F.mse_loss(sf1_pred, sf_target)
        loss_sf2 = F.mse_loss(sf2_pred, sf_target)

        train_store.update({
            'Qf1Mean': sf1_pred.mean(),
            'Qf2Mean': sf2_pred.mean(),
            'QfTargetsMean': sf_target.mean(),
            'QfTdErrsMean': ((sf_target - sf1_pred).mean() + (sf_target - sf2_pred).mean()) / 2,
            'LossQf1': loss_sf1,
            'LossQf2': loss_sf2,
        })

        mini_batch.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_targets(self) -> None:
        """Update successor feature networks.
        """
        target_sfs = [self.target_qf1, self.target_qf2]
        sfs = [self.qf1, self.qf2]
        for target_sf, sf in zip(target_sfs, sfs):
            for t_param, param in zip(target_sf.parameters(), sf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)
