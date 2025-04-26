#!/usr/bin/env python3
import sys
from typing import Any

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import argparse
import datetime
import functools
import os
import platform
import torch.multiprocessing as mp
import wandb

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garaged.src.garage.torch.modules import MLPModule

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.q_functions.discrete_mlp_q_function import DiscreteMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from garagei.envs.child_policy_env import ChildPolicyEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx
from garagei.torch.modules.categorical_mlp_module_ex import CategoricalMLPModuleEx

from iod.metra import METRA
from iod.metra_sf import MetraSf
from iod.dads import DADS
from iod.ppo import PPO
from iod.cic import CIC
from iod.sac import SAC
from iod.utils import get_normalizer_preset, get_gaussian_module_construction, get_log_dir, get_exp_name


EXP_DIR = 'exp'
if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'

def make_env(args: argparse.Namespace, max_path_length: int) -> Any:
    """Build environment.

    Args:
        args (argparse.Namespace): argparse arguments
        max_path_length (int): specifies the maximum number of timesteps of a single rollout

    Returns:
        Any: specified environmemt to use
    """
    if args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)

    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100, model_path='ant.xml')

    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  # Only support pixel-based environments
        if 'dmc_quadruped' in args.env:
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif 'dmc_humanoid' in args.env:
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError

        if args.env in ['dmc_quadruped_goal', 'dmc_humanoid_goal']:
            from envs.custom_dmc_tasks.goal_wrappers import GoalWrapper

            env = GoalWrapper(
                env,
                max_path_length=max_path_length,
                goal_range=args.goal_range,
                num_goal_steps=args.downstream_num_goal_steps,
            )
            cp_num_truncate_obs = 2

    elif args.env.startswith('robobin'):
        sys.path.append('lexa')
        from envs.lexa.robobin import MyRoboBinEnv
        if args.env == 'robobin':
            env = MyRoboBinEnv(log_per_goal=True)
        elif args.env == 'robobin_image':
            env = MyRoboBinEnv(obs_type='image', log_per_goal=True)

    elif args.env == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)

    elif args.env == 'ant_nav_prime':
        from envs.mujoco.ant_nav_prime_env import AntNavPrimeEnv

        env = AntNavPrimeEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            num_goal_steps=args.downstream_num_goal_steps,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2

    elif args.env == 'half_cheetah_goal':
        from envs.mujoco.half_cheetah_goal_env import HalfCheetahGoalEnv
        env = HalfCheetahGoalEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 1

    elif args.env == 'half_cheetah_hurdle':
        from envs.mujoco.half_cheetah_hurdle_env import HalfCheetahHurdleEnv

        env = HalfCheetahHurdleEnv(
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2

    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    # Potentially add normalization wrappers
    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        additional_dim = 0
        if args.env in ['ant_nav_prime']:
            normalizer_name = 'ant'
            additional_dim = cp_num_truncate_obs
        elif args.env in ['half_cheetah_goal', 'half_cheetah_hurdle']:
            normalizer_name = 'half_cheetah'
            additional_dim = cp_num_truncate_obs
        else:
            normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        if additional_dim > 0:
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    # Add child policy environment in case of hierarchical control
    if args.cp_path is not None:
        cp_path = args.cp_path
        if not os.path.exists(cp_path):
            import glob
            cp_path = glob.glob(cp_path)[0]
        cp_dict = torch.load(cp_path, map_location='cpu')

        env = ChildPolicyEnv(
            env,
            cp_dict,
            cp_action_range=1.5,
            cp_unit_length=args.cp_unit_length,
            cp_multi_step=args.cp_multi_step,
            cp_num_truncate_obs=cp_num_truncate_obs,
        )

    return env

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Logging
    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    parser.add_argument('--n_epochs_per_log', type=int, default=25)
    parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_random_trajectories', type=int, default=48)
    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')

    # Preprocessing
    parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--frame_stack', type=int, default=None)
    parser.add_argument('--video_skip_frames', type=int, default=1)

    # Architecture
    parser.add_argument('--encoder', type=int, default=0, help="Only used for image-based environments, where we add a simple CNN encoder to encode the image.")
    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_layer_norm', default=0, type=int, choices=[0, 1], help="Adds layer normalization to the Gaussian modules.")

    # Environment
    parser.add_argument('--max_path_length', type=int, default=200, help="Specifies the maximum number of timesteps of a single rollout.")
    parser.add_argument('--env', type=str, default='maze', choices=[
        # Exploration & goal reaching environments
        'ant', 'half_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'kitchen', 'robobin_image',
        # Hierarchical control environments
        'ant_nav_prime', 'half_cheetah_hurdle', 'half_cheetah_goal', 'dmc_quadruped_goal', 'dmc_humanoid_goal'
    ])

    # Training
    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--n_thread', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--traj_batch_size', type=int, default=8, help="Specifies how many trajectories of data collection to do before updating the policy.")
    parser.add_argument('--trans_minibatch_size', type=int, default=256, help="Specifies the batch size of the sample from the replay buffer.")
    parser.add_argument('--trans_optimization_epochs', type=int, default=200, help="Specifies how many updates to perform after each data collection round.")
    parser.add_argument('--common_lr', type=float, default=1e-4, help="Default learning rate for all neural networks.")
    parser.add_argument('--lr_op', type=float, default=None, help="Potential overwrite for the learning rate for the policy parameters.")
    parser.add_argument('--lr_te', type=float, default=None, help="Potential overwrite for the learning rate for the trajectory encoder parameters.")

    # General algorithmic parameters
    parser.add_argument('--dim_option', type=int, default=2, help="Specifies the skill dimension.")
    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1], help="Specifies whether to use discrete or continuous skills.")
    parser.add_argument('--alpha', type=float, default=0.01, help="Specifies the entropy coefficient (initial value if adaptive).")
    parser.add_argument('--algo', type=str, default='metra', choices=[
        # CSF (our method) & skill discovery baseliens
        'metra', 'metra_sf', 'dads', 'cic',
        # Hierarchical control algorithms
        'sac', 'ppo',
    ], help="Specifies the algorithm to use for training.")
    parser.add_argument('--sac_tau', type=float, default=5e-3, help="Specifies exponential moving average coefficient.")
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99, help="Specifies the discount factor.")
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=300000)
    parser.add_argument('--use_discrete_sac', type=int, default=0, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1], help="Whether to use Gaussian (0) or vMF (1) skills. This is only relevant for continuous skills.")
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1], help="Differentiates between transitions logic (METRA) and states logic (DIAYN)")
    parser.add_argument('--turn_off_dones', type=int, default=0, choices=[0, 1], help="This turns off any done=True flags from the environment and makes them done=False. See https://arxiv.org/pdf/1712.00378 for a detailed discussion.")

    # DADS specific parameters
    parser.add_argument('--num_alt_samples', type=int, default=100, help="(DADS only) Number of alternative z's to sample to compute skill dynamics denominator.")
    parser.add_argument('--split_group', type=int, default=65536, help="(DADS only)")
    parser.add_argument('--uniform_z', default=0, type=int, choices=[0, 1])

    # METRA specific parameters
    parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one'])
    parser.add_argument('--dual_lr', type=float, default=None)
    parser.add_argument('--fixed_lam', type=float, default=None)

    # VISR specific parameters
    parser.add_argument('--self_normalizing', type=int, default=0, choices=[0, 1])
    parser.add_argument('--no_diff_in_rep', default=0, type=int, choices=[0, 1])

    # CSF specific parameters
    parser.add_argument('--log_sum_exp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample_new_z', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_negative_z', type=int, default=256)
    parser.add_argument('--infonce_lam', type=float, default=1.0)

    # DIAYN specific parameters
    parser.add_argument('--diayn_include_baseline', type=int, default=0, choices=[0, 1])

    # Ablation parameters
    parser.add_argument('--add_log_sum_exp_to_rewards', type=int, default=0, choices=[0, 1], help="Adds the second term in the contrastive loss (the 'log-sum-exp') to the rewards as well")
    parser.add_argument('--add_penalty_to_rewards', default=0, type=int, choices=[0, 1])
    parser.add_argument('--metra_mlp_rep', type=int, default=0, choices=[0, 1], help="Uses the more general f(s, s')^T z parametrization.")

    # Zero-shot goal reaching flags
    parser.add_argument('--goal_range', type=float, default=50)
    parser.add_argument('--num_zero_shot_goals', type=int, default=50)
    parser.add_argument('--eval_goal_metrics', type=int, default=0, choices=[0, 1], help="Whether to evaluate zero-shot goal reaching during training.")

    # Hierarchical control flags
    parser.add_argument('--cp_path', type=str, default=None)
    parser.add_argument('--cp_path_idx', type=int, default=None)  # For exp name
    parser.add_argument('--cp_multi_step', type=int, default=1)
    parser.add_argument('--cp_unit_length', type=int, default=0)
    parser.add_argument('--downstream_reward_type', type=str, default='esparse')
    parser.add_argument('--downstream_num_goal_steps', type=int, default=50)
    parser.add_argument('--policy_type', type=str, default='gaussian', choices=['gaussian', 'categorical'])

    # CIC specific parameters
    parser.add_argument('--cic_temp', type=float, default=0.5)
    parser.add_argument('--cic_alpha', type=float, default=1.0)
    parser.add_argument('--apt_knn_k', type=int, default=16)
    parser.add_argument('--apt_use_icm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--apt_rms', type=int, default=1, choices=[0, 1])

    # LSD & CIC specific parameters
    parser.add_argument('--alive_reward', type=float, default=None)
    parser.add_argument('--dual_dist_scaling', type=str, default='geom', choices=['none', 'geom'])
    parser.add_argument('--const_scaler', type=float, default=1.)
    parser.add_argument('--wdm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_cpc', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_idz', type=int, default=1, choices=[0, 1])
    parser.add_argument('--wdm_ids', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_diff', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--joint_train', type=int, default=1, choices=[0, 1])

    return parser


args = get_argparser().parse_args()
g_start_time = int(datetime.datetime.now().timestamp())

@wrap_experiment(log_dir=get_log_dir(args, g_start_time, EXP_DIR), name=get_exp_name(args, g_start_time)[0])
def run(ctxt=None):
    # Print argparse arguments
    dowel.logger.log('ARGS: ' + str(args))

    # Set number of threads
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    # Fix seed
    set_seed(args.seed)

    # Setup runner
    runner = OptionLocalRunner(ctxt)

    # Setup environment
    max_path_length = args.max_path_length
    if args.cp_path is not None:
        max_path_length *= args.cp_multi_step
    contextualized_make_env = functools.partial(make_env, args=args, max_path_length=max_path_length)
    env = contextualized_make_env()
    example_ob = env.reset()

    # If using pixel-based environments, we need to know the pixel shape
    if args.encoder:
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        else:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None

    # Setup device
    device = torch.device('cuda' if args.use_gpu else 'cpu')

    # Hidden sizes for all following networks
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    # Determine link function
    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    # Get observation and action dimensions
    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    # If using pixel-based environments, we need to know the obs dimension *after* encoding
    if args.encoder:
        def make_encoder(**kwargs):
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                kwargs = {}
                encoder = make_encoder(**kwargs)

            return WithEncoder(encoder=encoder, module=module)

        kwargs = {}
        example_encoder = make_encoder(**kwargs)
        module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        module_obs_dim = obs_dim

    # ********************
    # Setup policy network
    # ********************
    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )
    module_kwargs = dict(
        hidden_sizes=master_dims,
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    if args.use_discrete_sac:
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    elif args.policy_type == 'categorical':
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    else:
        module_cls = GaussianMLPTwoHeadedModuleEx
        module_kwargs.update(dict(
            max_std=np.exp(2.),
            normal_distribution_cls=TanhNormal,
            output_w_init=functools.partial(xavier_normal_ex, gain=1.),
            init_std=1.,
        ))

    policy_q_input_dim = module_obs_dim + args.dim_option

    if args.algo in ['sac', 'ppo']:
        policy_q_input_dim = module_obs_dim

    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )
    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs)

    # ************************
    # Setup trajectory encoder
    # ************************
    output_dim = args.dim_option
    traj_encoder_obs_dim = module_obs_dim
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=traj_encoder_obs_dim,
        output_dim=output_dim,
        layer_normalization=args.use_layer_norm
    )
    traj_encoder = module_cls(**module_kwargs)
    if args.encoder:
        if args.spectral_normalization:
            te_encoder = make_encoder(spectral_normalization=True)
        else:
            te_encoder = None
        traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)


    # ********************
    # Setup dist predictor
    # ********************
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    # Setup dual lambda
    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    # ********************
    # Setup skill dynamics
    # ********************
    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )
    if args.algo == 'dads':
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    # For CIC, also setup prediction network and z encoder
    if args.algo == 'cic':
        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option * 2,
            output_dim=args.dim_option,
            layer_normalization=False, #args.phi_layer_norm,
        )
        pred_net = module_cls(**module_kwargs)

        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option,
            output_dim=output_dim,
            layer_normalization=False, #args.phi_layer_norm,
            spectral_normalization=True,
        )
        z_encoder = module_cls(**module_kwargs)
    else:
        pred_net = None

    def _finalize_lr(lr: float) -> float:
        """Post-process learning rate.

        Args:
            lr (float): initial lr, which can be None

        Returns:
            float: final lr after processing
        """
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    # If using prediction network, add its parameters to the optimizer
    if pred_net is not None:
        te_params = list(traj_encoder.parameters()) + list(z_encoder.parameters()) + list(pred_net.parameters())
    else:
        te_params = list(traj_encoder.parameters())

    # ****************
    # Setup optimizers
    # ****************
    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': te_params, 'lr': _finalize_lr(args.lr_te)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args.dual_lr)},
        ]),
    }

    if skill_dynamics is not None:
        optimizers.update({
            'skill_dynamics': torch.optim.Adam([
                {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    if dist_predictor is not None:
        optimizers.update({
            'dist_predictor': torch.optim.Adam([
                {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    # Setup replay buffer
    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(args.sac_max_buffer_size),
        pixel_shape=pixel_shape,
        sample_goals=(args.algo == 'crl' or args.algo == 'metra_sf'),
        discount=args.sac_discount,
    )

    # ****************
    # Setup Q networks
    # ****************
    qf1 = None
    qf2 = None
    log_alpha = None
    if args.algo in ['metra', 'dads', 'sac', 'cic']:
        if args.use_discrete_sac:
            qf1 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf1 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        if args.encoder:
            qf1 = with_encoder(qf1)

        if args.use_discrete_sac:
            qf2 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf2 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        if args.encoder:
            qf2 = with_encoder(qf2)
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    # NOTE: for metra_sf, the q networks are really just the "psi" successor features that
    # are learned in the same way as the q functions in the other algorithms
    elif args.algo == 'metra_sf':
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)

        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)

        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    elif args.algo == 'ppo':
        # TODO: Currently not support pixel obs
        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=False,
        )
        optimizers.update({
            'vf': torch.optim.Adam([
                {'params': vf.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    # This is for the parametrization ablation, where we use the encoding f(s, s')^T z
    f_encoder = None
    if args.metra_mlp_rep:
        f_encoder = ContinuousMLPQFunctionEx(
            obs_dim=obs_dim,
            action_dim=obs_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )

        optimizers.update({
            'f_encoder': torch.optim.Adam([
                {'params': list(f_encoder.parameters()), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    # Combine all optimizers
    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log,
        n_epochs_per_tb=args.n_epochs_per_log,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
        dim_option=args.dim_option,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name=args.algo,
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,

        pixel_shape=pixel_shape
    )

    # *****************************
    # Setup the specified algorithm
    # *****************************
    if args.algo == 'metra':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = METRA(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'metra_sf':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = MetraSf(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'cic':
        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = CIC(
            **algo_kwargs,
            **skill_common_args,

            pred_net=pred_net,
            z_encoder=z_encoder,
            cic_temp=args.cic_temp,
            cic_alpha=args.cic_alpha,
            knn_k=args.apt_knn_k,
            rms=args.apt_rms,
            alive_reward=args.alive_reward,

            dual_dist_scaling=args.dual_dist_scaling,
            const_scaler=args.const_scaler,
            wdm=args.wdm,
            wdm_cpc=args.wdm_cpc,
            wdm_idz=args.wdm_idz,
            wdm_ids=args.wdm_ids,
            wdm_diff=args.wdm_diff,
            aug=args.aug,
            joint_train=args.joint_train,
        )
    elif args.algo == 'sac':
        algo_kwargs.update(
            use_discrete_sac=args.use_discrete_sac,
        )

        algo = SAC(
            **algo_kwargs,
            **skill_common_args
        )

    elif args.algo == 'dads':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
        )

        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = DADS(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )

    else:
        raise NotImplementedError

    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    wandb.init(
        project="identifiable-rl",
        config=vars(args),
    )

    # Setup runner
    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )
    algo.option_policy.to(device)

    # Start training
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)


if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    run()
