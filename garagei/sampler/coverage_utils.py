"""Helpers for coverage calculations shared between samplers."""

from __future__ import annotations

import numpy as np


_ROBOBIN_COORD_DIMS = slice(0, 9)

_KITCHEN_OBJECT_DIMS = slice(11, 30)

_COVERAGE_SPEC = [
    (("robobin", "robobin_image"), dict(dims=_ROBOBIN_COORD_DIMS, rounding=("round", 2))),
    (("kitchen",), dict(dims=_KITCHEN_OBJECT_DIMS, rounding=("round", 2))),
    (
        (
            "half_cheetah",
            "half_cheetah_goal",
            "half_cheetah_hurdle",
        ),
        dict(dims=[0], rounding=("floor", None)),
    ),
    (
        (
            "ant",
            "ant_nav_prime",
            "ant_goal",
            "ant_pixel",
        ),
        dict(dims=[0, 1], rounding=("floor", None)),
    ),
    (("dmc_cheetah",), dict(dims=[0], rounding=("floor", None))),
    (("dmc_quadruped",), dict(dims=[0, 1], rounding=("floor", None))),
    (("dmc_humanoid",), dict(dims=[0, 1], rounding=("floor", None))),
]

_DEFAULT_SPEC = dict(dims=None, rounding=("round", 2))


def _match_env_name(env_name, prefixes):
    if env_name is None:
        return False
    env_name = env_name.lower()
    for prefix in prefixes:
        if env_name == prefix or env_name.startswith(prefix + "_"):
            return True
    return False


def get_coverage_spec(env_name):
    """Return (dims, rounding) tuple for the given environment name."""
    for prefixes, spec in _COVERAGE_SPEC:
        if _match_env_name(env_name, prefixes):
            return spec
    return _DEFAULT_SPEC


def discretize_states_for_coverage(states, active_dims, env_name):
    """Project raw states onto the coordinates used in MjNumUniqueCoords."""
    if states is None:
        return None

    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    spec = get_coverage_spec(env_name)
    dims = spec["dims"]

    selected = None
    if dims is not None:
        try:
            selected = arr[:, dims]
        except Exception:
            selected = None

    if selected is None or selected.size == 0:
        if active_dims is not None and np.ndim(active_dims) == 1 and len(active_dims) == arr.shape[1]:
            selected = arr[:, active_dims]
        else:
            selected = arr

    rounding_mode, rounding_value = spec["rounding"]
    if rounding_mode == "floor":
        discretized = np.floor(selected)
    else:
        decimals = 2 if rounding_value is None else rounding_value
        discretized = np.round(selected, decimals=decimals)

    return discretized
