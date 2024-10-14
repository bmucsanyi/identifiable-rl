import math

import torch
import torch.nn.functional as F

from iod.lsd import LSD

from iod.utils import RMS


class CIC(LSD):
    def __init__(
            self,
            *,
            pred_net,

            cic_temp,
            cic_alpha,
            knn_k,
            rms,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.pred_net = pred_net.to(self.device)

        self.param_modules.update(
            pred_net=self.pred_net,
        )

        self.cic_temp = cic_temp
        self.cic_alpha = cic_alpha
        self.knn_k = knn_k
        self.knn_clip = 0.0005
        self.knn_avg = True
        self.rms = rms

        self.exp_knn_rms = RMS(self.device)

    def _compute_apt_reward(self, source, target):
        with torch.no_grad():
            b1, b2 = source.size(0), target.size(0)
            # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
            sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1,
                                    p=2)
            reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

            if not self.knn_avg:  # only keep k-th nearest neighbor
                reward = reward[:, -1]
                reward = reward.reshape(-1, 1)  # (b1, 1)
                if self.rms:
                    moving_mean, moving_std = self.exp_knn_rms(reward)
                    reward = reward / moving_std
                reward = torch.max(reward - self.knn_clip, torch.zeros_like(reward).to(self.device))  # (b1, )
            else:  # average over all k nearest neighbors
                reward = reward.reshape(-1, 1)  # (b1 * k, 1)
                if self.rms:
                    moving_mean, moving_std = self.exp_knn_rms(reward)
                    reward = reward / moving_std
                reward = torch.max(reward - self.knn_clip, torch.zeros_like(reward).to(self.device))
                reward = reward.reshape((b1, self.knn_k))  # (b1, k)
                reward = reward.mean(dim=1)  # (b1,)
            reward = torch.log(reward + 1.0)
        return reward

    def _update_rewards(self, tensors, v):
        assert not self.gt_reward

        if self.te_original_obs:
            obs = v['ori_obs']
            next_obs = v['next_ori_obs']
        else:
            obs = v['obs']
            next_obs = v['next_obs']

        source = self.traj_encoder(obs).mean
        target = self.traj_encoder(next_obs).mean
        with torch.no_grad():
            if self.cic_alpha > 0:
                apt_reward = self._compute_apt_reward(source, target)
            else:
                apt_reward = 0.

            if self.cic_alpha < 1:
                cic_reward, _ = self._compute_cpc_loss(obs, next_obs, v['options'])
                cic_reward = -cic_reward
            else:
                cic_reward = 0.
                
            rewards = self.cic_alpha * apt_reward + (1 - self.cic_alpha) * cic_reward

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        if self.alive_reward is not None:
            rewards = rewards + self.alive_reward

        v['rewards'] = rewards

    def _get_query_key(self, obs, next_obs, skill):
        state = self.traj_encoder(obs).mean
        next_state = self.traj_encoder(next_obs).mean
        query = self.z_encoder(skill).mean
        key = self.pred_net(torch.cat([state, next_state], 1)).mean
        return query, key

    def _compute_cpc_loss(self, obs, next_obs, options):
        temperature = self.cic_temp
        eps = 1e-6
        query, key = self._get_query_key(obs, next_obs, options)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)

        return loss, cov / temperature

    def _update_loss_te(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']
        obs = self._restrict_te_obs(obs)
        next_obs = self._restrict_te_obs(next_obs)

        loss, logits = self._compute_cpc_loss(obs, next_obs, v['options'])

        loss_te = loss.mean()

        tensors.update({
            'LossTe': loss_te,
            'CicLogits': logits.norm(),
        })

        return loss
