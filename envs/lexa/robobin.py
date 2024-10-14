import akro

from lexa.lexa_envs import RoboBinEnv
import numpy as np


class MyRoboBinEnv(RoboBinEnv):
    def __init__(self, obs_type='state', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_type = obs_type
        self.last_obs_dict = None
        self.last_obs = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.ob_info = dict(
            type='pixel',
            pixel_shape=(64, 64, 3),
        )

    @property
    def observation_space(self):
        if self.obs_type == 'state':
            return akro.Box(low=-np.inf, high=np.inf, shape=(9,))
        elif self.obs_type == 'image':
            return akro.Box(low=-np.inf, high=np.inf, shape=(64, 64, 3))
        else:
            raise NotImplementedError

    def get_state(self, obs_dict):
        if self.obs_type == 'state':
            state = obs_dict['state']
        elif self.obs_type == 'image':
            state = obs_dict['image'].flatten()
        return state

    def reset(self):
        obs_dict = super().reset()
        obs = self.get_state(obs_dict)
        self.last_obs_dict = obs_dict
        self.last_obs = obs
        return obs

    def step(self, action, render=False):
        obs_dict, reward, done, info = super().step(action)
        obs = self.get_state(obs_dict)

        # xyz of hand, obj1, and obj2
        coords = self.last_obs_dict['state'].copy()
        next_coords = obs_dict['state'].copy()
        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_obs_dict['state']
        info['next_ori_obs'] = obs_dict['state']
        info['image'] = obs_dict['image']
        info['image_goal'] = obs_dict['image_goal']
        info['success'] = obs_dict['metric_success/goal_{}'.format(self.get_goal_idx())]
        if render:
            info['render'] = obs_dict['image'].transpose(2, 0, 1)

        self.last_obs_dict = obs_dict
        self.last_obs = obs

        return obs, reward, done, info

    def plot_trajectory(self, trajectory, color, ax):
        # hand (x, y)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].dtype == object:
                coordinates_trajectories.append(np.concatenate([
                    np.concatenate(trajectory['env_infos']['coordinates'], axis=0),
                    [trajectory['env_infos']['next_coordinates'][-1][-1]],
                ]))
            elif trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))
            else:
                assert False
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None, unique_coord_decimals=2):
        # obj 1 and obj 2
        coord_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        eval_metrics = {}

        if coord_dims is not None:
            coords = []
            success = []
            for traj in trajectories:
                traj1 = traj['env_infos']['coordinates'][:, coord_dims]
                traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
                coords.append(traj1)
                coords.append(traj2)

                traj_success = traj['env_infos']['success']

                any_success = (np.sum(traj_success) >= 1).astype(float)
                success.append(any_success)
            coords = np.concatenate(coords, axis=0)
            uniq_coords = np.unique(np.round(coords, decimals=unique_coord_decimals), axis=0)
            success_rate = np.mean(success)
            eval_metrics.update({
                'MjNumTrajs': len(trajectories),
                'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
                'MjNumCoords': len(coords),
                'MjNumUniqueCoords': len(uniq_coords),
                'MjSuccessRate': success_rate,
            })

        return eval_metrics