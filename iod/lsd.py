import pathlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy
import os
import pickle

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians #, RandomShiftsAug


class LSD(IOD):
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
            alive_reward,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,
            dual_dist_scaling,
            const_scaler,

            wdm,
            wdm_cpc,
            wdm_idz,
            wdm_ids,
            wdm_diff,

            pixel_shape=None,
            aug=False,
            gt_reward=False,
            joint_train=False,

            goal_range=None,
            z_encoder=None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.z_encoder = z_encoder.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
            z_encoder=self.z_encoder,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.alive_reward = alive_reward
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist
        self.dual_dist_scaling = dual_dist_scaling
        self.const_scaler = const_scaler

        self.wdm = wdm
        self.wdm_cpc = wdm_cpc
        self.wdm_idz = wdm_idz
        self.wdm_ids = wdm_ids
        self.wdm_diff = wdm_diff
        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape
        self.aug = aug
        if self.aug:
            self.aug_module = RandomShiftsAug(pad=4)
        self.gt_reward = gt_reward
        self.joint_train = joint_train
        self.goal_range = goal_range

        self.te_restrict_obs_idxs = None
        self.te_original_obs = False

        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _restrict_te_obs(self, obs):
        if self.te_restrict_obs_idxs is not None:
            return obs[:, self.te_restrict_obs_idxs]
        return obs

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

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

    def _aug_data(self, data):
        if self.aug:
            for key in ['obs', 'next_obs']:
                image = data[key][..., :np.prod(self.pixel_shape)].reshape(-1, *self.pixel_shape).permute(0, 3, 1, 2)
                aug_image = self.aug_module(image)
                data[key] = torch.cat([aug_image.permute(0, 2, 3, 1).reshape(-1, np.prod(self.pixel_shape)), data[key][0][..., np.prod(self.pixel_shape):]], dim=-1)

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

        if self.joint_train:
            tensors = self._train_components(epoch_data)
        else:
            tensors = {}
            if not self.gt_reward:
                tensors.update(self._train_components(epoch_data, train_op=False, train_te=True, force_on_policy=self.te_on_policy))
            tensors.update(self._train_components(epoch_data, train_op=True, train_te=False))

        return tensors

    def _train_components(self, epoch_data, train_op=True, train_te=True, force_on_policy=False):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None or force_on_policy:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._aug_data(v)

            if train_te:
                self._optimize_te(tensors, v)
            if train_op:
                self._update_rewards(tensors, v)
                self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

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

    def _update_rewards(self, tensors, v):
        if self.gt_reward:
            rewards = v['rewards']
            tensors.update({
                'PureRewardMean': rewards.mean(),
                'PureRewardStd': rewards.std(),
            })
            return

        if self.te_original_obs:
            obs = v['ori_obs']
            next_obs = v['next_ori_obs']
        else:
            obs = v['obs']
            next_obs = v['next_obs']
        obs = self._restrict_te_obs(obs)
        next_obs = self._restrict_te_obs(next_obs)

        if self.wdm:
            if self.wdm_ids:
                if self.wdm_diff:
                    s_score = next_obs - obs
                else:
                    s_score = next_obs
            else:
                cur_z = self.traj_encoder(obs).mean
                next_z = self.traj_encoder(next_obs).mean
                if self.wdm_diff:
                    s_score = next_z - cur_z
                else:
                    s_score = next_z
                v.update({
                    'cur_z': cur_z,
                    'next_z': next_z,
                })
            if self.wdm_idz:
                z_score = v['options']
            else:
                z_score = self.z_encoder(v['options']).mean
            te_score = (s_score * z_score).sum(dim=1)

            alt_te_scores = s_score @ z_score.T

            if self.wdm_cpc:
                logits = alt_te_scores
                labels = torch.arange(logits.size(0), device=self.device)
                rewards = -F.cross_entropy(logits, labels, reduction='none') + np.log(logits.size(0))
            else:
                rewards = te_score - alt_te_scores.mean(dim=1)
        else:
            if self.inner:
                cur_z = self.traj_encoder(obs).mean
                next_z = self.traj_encoder(next_obs).mean
                target_z = next_z - cur_z

                if self.discrete:
                    masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                    rewards = (target_z * masks).sum(dim=1)
                else:
                    inner = (target_z * v['options']).sum(dim=1)
                    rewards = inner

                # For dual LSD
                v.update({
                    'cur_z': cur_z,
                    'next_z': next_z,
                })
            else:
                target_dists = self.traj_encoder(next_obs)

                if self.discrete:
                    logits = target_dists.mean
                    rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
                else:
                    rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        if self.alive_reward is not None:
            rewards = rewards + self.alive_reward

        v['rewards'] = rewards

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        if self.te_original_obs:
            obs = v['ori_obs']
            next_obs = v['next_ori_obs']
        else:
            obs = v['obs']
            next_obs = v['next_obs']
        obs = self._restrict_te_obs(obs)
        next_obs = self._restrict_te_obs(next_obs)

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0]) * self.const_scaler
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                if self.dual_dist_scaling == 'none':
                    scaling_factor = 1. / (s2_dist_std ** 2)
                    normalized_scaling_factor = scaling_factor
                else:
                    scaling_factor = 1. / s2_dist_std
                    geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                    normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _evaluate_policy(self, runner):
        # Random trajectories
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.num_random_trajectories):
                random_options.append(eye_options[i % self.dim_option])
                colors.append(i % self.dim_option)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_initial_state=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(self._restrict_te_obs(last_obs))

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()  # Plot from means

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        # Goal-conditioned metrics
        eval_option_metrics = {}
        if self.algo == 'lsd' and not self.wdm and self.env_name in ['kitchen', 'dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
            env = runner._env
            goals = []  # list of (goal_obs, goal_info)
            goal_metrics = defaultdict(list)
            if self.env_name == 'kitchen':
                goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
                for i, goal_name in enumerate(goal_names):
                    goal_obs = env.render_goal(goal_idx=i).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_idx': i, 'goal_name': goal_name}))
            elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid']:
                for i in range(20):
                    env.reset()
                    state = env.physics.get_state().copy()
                    if self.env_name == 'dmc_cheetah':
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                    env.physics.set_state(state)
                    if self.env_name == 'dmc_humanoid':
                        for _ in range(50):
                            env.step(np.zeros_like(env.action_space.sample()))
                    else:
                        env.step(np.zeros_like(env.action_space.sample()))
                    goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_loc': goal_loc}))
            elif self.env_name in ['ant', 'ant_pixel', 'half_cheetah']:
                for i in range(20):
                    env.reset()
                    state = env.unwrapped._get_obs().copy()
                    if self.env_name in ['half_cheetah']:
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                        env.set_state(state[:9], state[9:])
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                        env.set_state(state[:15], state[15:])
                    for _ in range(5):
                        env.step(np.zeros_like(env.action_space.sample()))
                    if self.env_name == 'ant_pixel':
                        goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                        goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    else:
                        goal_obs = env._apply_normalize_obs(state).astype(np.float32)
                    goals.append((goal_obs, {'goal_loc': goal_loc}))

            if self.unit_length:
                mean_length = 1.
            else:
                mean_length = np.linalg.norm(np.random.randn(1000000, self.dim_option), axis=1).mean()
            for method in ['Single', 'Adaptive'] if self.discrete else ['']:
                for goal_obs, goal_info in goals:
                    obs = env.reset()
                    step = 0
                    done = False
                    success = 0
                    option = None
                    while step < self.max_path_length and not done:
                        if self.inner:
                            te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(self.device)
                            phi_s, phi_g = self.traj_encoder(self._restrict_te_obs(te_input)).mean
                            phi_s, phi_g = phi_s.detach().cpu().numpy(), phi_g.detach().cpu().numpy()
                            if self.discrete:
                                if method == 'Adaptive':
                                    option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                                else:
                                    if option is None:
                                        option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                            else:
                                option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s) * mean_length
                        else:
                            te_input = torch.from_numpy(goal_obs[None, ...]).to(self.device)
                            phi = self.traj_encoder(self._restrict_te_obs(te_input)).mean[0]
                            phi = phi.detach().cpu().numpy()
                            if self.discrete:
                                option = np.eye(self.dim_option)[phi.argmax()]
                            else:
                                option = phi
                        action, agent_info = self.option_policy.get_action(np.concatenate([obs, option]))
                        next_obs, _, done, info = env.step(action)
                        obs = next_obs

                        if self.env_name == 'kitchen':
                            success = max(success, env.compute_success(goal_info['goal_idx'])[0])

                        step += 1

                    if self.env_name == 'kitchen':
                        goal_metrics[f'Kitchen{method}Goal{goal_info["goal_name"]}'].append(success)
                        goal_metrics[f'Kitchen{method}GoalOverall'].append(success * len(goal_names))
                    elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                        if self.env_name in ['dmc_cheetah']:
                            cur_loc = env.physics.get_state()[:1]
                        elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                            cur_loc = env.physics.get_state()[:2]
                        elif self.env_name in ['half_cheetah']:
                            cur_loc = env.unwrapped._get_obs()[:1]
                        else:
                            cur_loc = env.unwrapped._get_obs()[:2]
                        distance = np.linalg.norm(cur_loc - goal_info['goal_loc'])
                        squared_distance = distance ** 2
                        goal_metrics[f'Goal{method}Distance'].append(distance)
                        goal_metrics[f'Goal{method}SquaredDistance'].append(squared_distance)

            goal_metrics = {key: np.mean(value) for key, value in goal_metrics.items()}
            eval_option_metrics.update(goal_metrics)

        # Train coverage metric
        # if len(self.coverage_queue) > 0:
        #     coverage_data = np.array(self.coverage_queue)
        #     if self.env_name == 'kitchen':
        #         coverage = coverage_data.max(axis=0)
        #         goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']

        #         for i, goal_name in enumerate(goal_names):
        #             eval_option_metrics[f'TrainKitchenTask{goal_name}'] = coverage[i]
        #         eval_option_metrics[f'TrainKitchenOverall'] = coverage.sum()
        #     else:
        #         uniq_coords = np.unique(np.floor(coverage_data), axis=0)
        #         eval_option_metrics['TrainNumUniqueCoords'] = len(uniq_coords)
        #         self.coverage_log = list(np.unique(self.coverage_log, axis=0))
        #         eval_option_metrics['TrainTotalNumUniqueCoords'] = len(self.coverage_log)
        # else:
        #     if self.env_name == 'kitchen':
        #         goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
        #         for i, goal_name in enumerate(goal_names):
        #             eval_option_metrics[f'TrainKitchenTask{goal_name}'] = 0
        #         eval_option_metrics[f'TrainKitchenOverall'] = 0
        #     else:
        #         eval_option_metrics['TrainNumUniqueCoords'] = 0
        #         eval_option_metrics['TrainTotalNumUniqueCoords'] = 0

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_initial_state=False,
                    _deterministic_policy=self.eval_deterministic_video,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)

        # if self.eval_for_plot == 1:
        #     eval_info = []
        #     states = []
        #     for traj in random_trajectories:
        #         ori_obs = np.concatenate([traj['env_infos']['original_observations'], traj['env_infos']['original_next_observations'][-1:]], axis=0)
        #         states.append(ori_obs)
        #     eval_info.append((random_options, states))

        #     file_name = os.path.join(runner._snapshotter._snapshot_dir, 'eval_info.pkl')
        #     with open(file_name, 'wb') as f:
        #         pickle.dump(eval_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        #     exit(0)

        # if self.pixel_shape is not None:
        #     for traj in random_trajectories:
        #         traj['observations'] = traj['next_observations'] = None
        #         traj['env_infos']['original_observations'] = traj['env_infos']['original_next_observations'] = None
        # path = pathlib.Path(runner._snapshotter._snapshot_dir) / 'eval_trajs' / f'{runner.step_itr}.pkl'
        # path.parent.mkdir(parents=True, exist_ok=True)
        # with open(path, 'wb') as f:
        #     pickle.dump({
        #         'random_trajectories': random_trajectories,
        #         'random_options': random_options,
        #     }, f, protocol=pickle.HIGHEST_PROTOCOL)
