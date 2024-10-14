import akro
import numpy as np
from envs.mujoco.ant_env import AntEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym import utils


class AntNavPrimeEnv(AntEnv):
    def __init__(
            self,
            max_path_length,
            goal_range,
            num_goal_steps,
            reward_type='sparse',
            **kwargs,
    ):
        self.max_path_length = max_path_length
        self.reward_type = reward_type

        self.goal_epsilon = 3.
        self.goal_range = goal_range
        self.num_goal_steps = num_goal_steps

        self.goals = [np.random.uniform(-self.goal_range, self.goal_range, (2,))]
        for i in range(3):
            _x, _y = self.goals[-1]
            self.goals.append(np.array([
                np.random.uniform(_x - self.goal_range, _x + self.goal_range),
                np.random.uniform(_y - self.goal_range, _y + self.goal_range)
            ]))

        self.cur_goal = self.goals[0]
        self.num_steps = 0
        self.goal_success = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_staying = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_idx = 1

        super().__init__(**kwargs)
        utils.EzPickle.__init__(self, max_path_length=max_path_length, goal_range=goal_range, num_goal_steps=num_goal_steps, reward_type=reward_type, **kwargs)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((2,), -float('inf'), dtype=np.float32)
        high = np.full((2,), float('inf'), dtype=np.float32)
        return akro.concat(self.observation_space, akro.Box(low=low, high=high, dtype=self.observation_space.dtype))

    def reset_model(self):
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.num_steps = 0
        self.goal_idx = 1
        self.goal_success = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_staying = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, self.cur_goal])

        return obs

    def step(self, *args, **kwargs):
        ob, reward, done, info = super().step(*args, **kwargs)
        for k in self.goal_success:
            info[k] = self.goal_success[k]
            info[f'{k}_staying'] = self.goal_staying[k]

        return ob, reward, done, info

    def _get_done(self):
        return self.num_steps == self.max_path_length

    def compute_reward(self, xposbefore, yposbefore, xposafter, yposafter):
        self.num_steps += 1
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter, yposafter]))
        if self.reward_type == 'sparse':
            if self.num_steps % self.num_goal_steps == 0:
                reward = -delta
            else:
                reward = 0.
        elif self.reward_type == 'esparse':
            if self.num_steps != 1 and delta <= self.goal_epsilon:
                # reward = 10. / (self.max_path_length / self.num_goal_steps)
                reward = 1.0
                self.goal_success[f'goal_{self.goal_idx}'] = 1
                self.goal_staying[f'goal_{self.goal_idx}'] += 1
                # NOTE: this skips the timesteps once we reach the goal
                # self.num_steps = (self.num_steps + self.num_goal_steps - 1) // self.num_goal_steps * self.num_goal_steps
            else:
                reward = -0.
        elif self.reward_type == 'ddense':
            delta_before = np.linalg.norm(self.cur_goal - np.array([xposbefore, yposbefore]))
            reward = delta_before - delta
        elif self.reward_type == 'dense':
            reward = -delta / self.max_path_length

        if self.num_steps % self.num_goal_steps == 0:
            # self.cur_goal = np.array([
            #     np.random.uniform(xposafter - self.goal_range, xposafter + self.goal_range),
            #     np.random.uniform(yposafter - self.goal_range, yposafter + self.goal_range),
            # ])
            self.goal_idx += 1
            self.cur_goal = self.goals[(self.goal_idx - 1) % 4]


        return reward
