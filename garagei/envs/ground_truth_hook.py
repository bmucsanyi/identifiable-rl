import gym
import numpy as np


def _find_env_with_attr(env, attr_name):
    """Traverse wrappers until an environment exposing attr_name is found."""
    current = env
    while True:
        if hasattr(current, attr_name):
            return current
        if isinstance(current, gym.Wrapper):
            current = current.env
        else:
            break
    return current if hasattr(current, attr_name) else None


def _mujoco_full_state_extractor(env, *_):
    """Return concatenated (qpos, qvel) from the first Mujoco env in the stack."""
    target = _find_env_with_attr(env, 'sim')
    if target is None:
        return None
    data = target.sim.data
    return np.concatenate([data.qpos.copy(), data.qvel.copy()], axis=0)


def _passthrough_ground_truth_extractor(env, *_):
    """Read ground_truth_state from the first env exposing it in the stack."""
    target = _find_env_with_attr(env, 'ground_truth_state')
    if target is None:
        return None
    state = getattr(target, 'ground_truth_state', None)
    return None if state is None else state


class GroundTruthHookWrapper(gym.Wrapper):
    """Wrapper that keeps a copy of ground-truth states via a user-provided hook."""

    def __init__(self, env, extractor):
        super().__init__(env)
        self._extractor = extractor
        self.ground_truth_state = None

        # Preserve action/observation spaces for downstream wrappers.
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _update_ground_truth_state(self, obs, info):
        state = self._extractor(self.env, obs, info)
        if state is not None:
            self.ground_truth_state = np.array(state, copy=True)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._update_ground_truth_state(obs, info=None)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        self._update_ground_truth_state(obs, info)
        return obs, reward, done, info


def maybe_wrap_with_ground_truth_hook(env, env_name):
    """Attach a ground-truth hook for specific environments."""
    extractor = None
    if env_name.startswith('dmc'):
        extractor = _passthrough_ground_truth_extractor
    elif env_name.startswith('half_cheetah') or env_name.startswith('ant'):
        extractor = _mujoco_full_state_extractor

    if extractor is None:
        return env
    return GroundTruthHookWrapper(env, extractor)
