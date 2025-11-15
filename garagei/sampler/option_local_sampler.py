"""Sampler that runs workers in the main process."""
import copy
from collections import defaultdict
import numpy as np

from garage import TrajectoryBatch
from garage.sampler import LocalSampler

from iod.disentanglement import linear_disentanglement

STEP_SIZES = [1, 2, 3, 4, 5, 10, 20]
VARIANCE_THRESHOLD = 1e-8


def aggregate_sac_metrics(sac_payloads, metra_states_list):
    """Aggregate SAC evaluation metrics using payloads from workers."""
    payloads = [payload for payload in sac_payloads if payload]
    if not payloads:
        return {}

    metra_states = None
    filtered_states = [states for states in metra_states_list if states is not None]
    if filtered_states:
        metra_states = np.concatenate(filtered_states, axis=0)

    metrics = {}
    seen_keys = set()
    for payload in payloads:
        env_name = payload.get("env_name")
        sac_dir = payload.get("sac_states_dir")
        for entry in payload.get("sac_states", []):
            step = entry.get("step")
            key = (sac_dir, env_name, step)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            metrics.update(_compute_sac_metrics_for_entry(env_name, entry, metra_states))
    return metrics


def _compute_sac_metrics_for_entry(env_name, entry, metra_states):
    metrics = {}
    ground_truth_matrix = entry["ground_truth_matrix"]
    encoder_matrix = entry["encoder_matrix"]
    step_str = entry["step"]

    ground_truth_matrix = np.asarray(ground_truth_matrix, dtype=np.float64)
    encoder_matrix = np.asarray(encoder_matrix, dtype=np.float64)

    state_variance = np.var(ground_truth_matrix, axis=0)
    active_dims = state_variance > VARIANCE_THRESHOLD

    if np.sum(active_dims) > 0:
        ground_truth_filtered = ground_truth_matrix[:, active_dims]
        r_square = linear_disentanglement(ground_truth_filtered, encoder_matrix, mode="r2")
        pearson = linear_disentanglement(ground_truth_filtered, encoder_matrix, mode="pearson")
    else:
        r_square = float("nan")
        pearson = float("nan")

    metrics[f"r_square_sac_{step_str}"] = float(r_square)
    metrics[f"pearson_sac_{step_str}"] = float(pearson)

    for step_size in STEP_SIZES:
        r2_diff, pearson_diff = _compute_multi_step_metrics(ground_truth_matrix, encoder_matrix, step_size)
        metrics[f"r_square_diff_{step_size}_step_sac_{step_str}"] = float(r2_diff)
        metrics[f"pearson_diff_{step_size}_step_sac_{step_str}"] = float(pearson_diff)

    ground_truth_matrix_object = _extract_object_ground_truth(env_name, ground_truth_matrix)
    if ground_truth_matrix_object is not None:
        object_state_variance = np.var(ground_truth_matrix_object, axis=0)
        object_active_dims = object_state_variance > VARIANCE_THRESHOLD

        if np.sum(object_active_dims) > 0:
            ground_truth_object_filtered = ground_truth_matrix_object[:, object_active_dims]
            r_square_object = linear_disentanglement(ground_truth_object_filtered, encoder_matrix, mode="r2")
            pearson_object = linear_disentanglement(ground_truth_object_filtered, encoder_matrix, mode="pearson")
        else:
            r_square_object = float("nan")
            pearson_object = float("nan")

        metrics[f"r_square_sac_{step_str}_object"] = float(r_square_object)
        metrics[f"pearson_sac_{step_str}_object"] = float(pearson_object)

        for step_size in STEP_SIZES:
            r2_diff_object, pearson_diff_object = _compute_multi_step_metrics(
                ground_truth_matrix_object, encoder_matrix, step_size
            )
            metrics[f"r_square_diff_{step_size}_step_sac_{step_str}_object"] = float(r2_diff_object)
            metrics[f"pearson_diff_{step_size}_step_sac_{step_str}_object"] = float(pearson_diff_object)

    covered_mask, uncovered_mask = _compute_coverage_masks(metra_states, ground_truth_matrix, active_dims)
    num_covered = int(np.sum(covered_mask)) if covered_mask is not None else 0
    num_uncovered = int(np.sum(uncovered_mask)) if uncovered_mask is not None else 0
    metrics[f"num_covered_states_sac_{step_str}"] = num_covered
    metrics[f"num_uncovered_states_sac_{step_str}"] = num_uncovered

    # Covered subset metrics
    if num_covered > 0:
        ground_truth_covered = ground_truth_matrix[covered_mask]
        encoder_covered = encoder_matrix[covered_mask]
        metrics.update(
            _compute_subset_metrics(
                ground_truth_covered, encoder_covered, step_str, suffix="sac_covered"
            )
        )

        if ground_truth_matrix_object is not None:
            ground_truth_covered_object = ground_truth_matrix_object[covered_mask]
            metrics.update(
                _compute_subset_metrics(
                    ground_truth_covered_object,
                    encoder_covered,
                    step_str,
                    suffix="sac_covered",
                    object_suffix="_object",
                )
            )
    else:
        metrics.update(_nan_subset_metrics(step_str, suffix="sac_covered"))
        if ground_truth_matrix_object is not None:
            metrics.update(_nan_subset_metrics(step_str, suffix="sac_covered", object_suffix="_object"))

    # Uncovered subset metrics
    if num_uncovered > 0:
        ground_truth_uncovered = ground_truth_matrix[uncovered_mask]
        encoder_uncovered = encoder_matrix[uncovered_mask]
        metrics.update(
            _compute_subset_metrics(
                ground_truth_uncovered, encoder_uncovered, step_str, suffix="sac_uncovered"
            )
        )

        if ground_truth_matrix_object is not None:
            ground_truth_uncovered_object = ground_truth_matrix_object[uncovered_mask]
            metrics.update(
                _compute_subset_metrics(
                    ground_truth_uncovered_object,
                    encoder_uncovered,
                    step_str,
                    suffix="sac_uncovered",
                    object_suffix="_object",
                )
            )
    else:
        metrics.update(_nan_subset_metrics(step_str, suffix="sac_uncovered"))
        if ground_truth_matrix_object is not None:
            metrics.update(_nan_subset_metrics(step_str, suffix="sac_uncovered", object_suffix="_object"))

    return metrics


def _compute_subset_metrics(ground_truth_matrix, encoder_matrix, step_str, suffix, object_suffix=""):
    metrics = {}
    subset_variance = np.var(ground_truth_matrix, axis=0)
    subset_active_dims = subset_variance > VARIANCE_THRESHOLD

    if np.sum(subset_active_dims) > 0:
        ground_truth_filtered = ground_truth_matrix[:, subset_active_dims]
        r_square = linear_disentanglement(ground_truth_filtered, encoder_matrix, mode="r2")
        pearson = linear_disentanglement(ground_truth_filtered, encoder_matrix, mode="pearson")
    else:
        r_square = float("nan")
        pearson = float("nan")

    metrics[f"r_square_{suffix}_{step_str}{object_suffix}"] = float(r_square)
    metrics[f"pearson_{suffix}_{step_str}{object_suffix}"] = float(pearson)

    for step_size in STEP_SIZES:
        r2_diff, pearson_diff = _compute_multi_step_metrics(ground_truth_matrix, encoder_matrix, step_size)
        metrics[f"r_square_diff_{step_size}_step_{suffix}_{step_str}{object_suffix}"] = float(r2_diff)
        metrics[f"pearson_diff_{step_size}_step_{suffix}_{step_str}{object_suffix}"] = float(pearson_diff)

    return metrics


def _nan_subset_metrics(step_str, suffix, object_suffix=""):
    metrics = {}
    metrics[f"r_square_{suffix}_{step_str}{object_suffix}"] = float("nan")
    metrics[f"pearson_{suffix}_{step_str}{object_suffix}"] = float("nan")
    for step_size in STEP_SIZES:
        metrics[f"r_square_diff_{step_size}_step_{suffix}_{step_str}{object_suffix}"] = float("nan")
        metrics[f"pearson_diff_{step_size}_step_{suffix}_{step_str}{object_suffix}"] = float("nan")
    return metrics


def _compute_multi_step_metrics(ground_truth_matrix, encoder_matrix, step_size):
    if len(ground_truth_matrix) > step_size:
        gt_diff = ground_truth_matrix[step_size:] - ground_truth_matrix[:-step_size]
        enc_diff = encoder_matrix[step_size:] - encoder_matrix[:-step_size]

        diff_variance = np.var(gt_diff, axis=0)
        active_diff_dims = diff_variance > VARIANCE_THRESHOLD

        if np.sum(active_diff_dims) > 0:
            gt_diff_filtered = gt_diff[:, active_diff_dims]
            r2_diff = linear_disentanglement(gt_diff_filtered, enc_diff, mode="r2")
            pearson_diff = linear_disentanglement(gt_diff_filtered, enc_diff, mode="pearson")
        else:
            r2_diff = float("nan")
            pearson_diff = float("nan")
    else:
        r2_diff = float("nan")
        pearson_diff = float("nan")

    return r2_diff, pearson_diff


def _extract_object_ground_truth(env_name, ground_truth_matrix):
    if env_name == "kitchen":
        return ground_truth_matrix[:, 11:30]
    if env_name in ["robobin", "robobin_image"]:
        return ground_truth_matrix[:, 3:9]
    return None


def _compute_coverage_masks(metra_states, ground_truth_matrix, active_dims):
    if metra_states is None or np.sum(active_dims) == 0:
        return None, None

    decimals = 2
    sac_discretized = np.round(ground_truth_matrix[:, active_dims], decimals=decimals)
    metra_discretized = np.round(metra_states[:, active_dims], decimals=decimals)

    metra_unique_set = set(map(tuple, metra_discretized))
    covered_mask = np.array([tuple(s) in metra_unique_set for s in sac_discretized])
    uncovered_mask = ~covered_mask
    return covered_mask, uncovered_mask



class OptionLocalSampler(LocalSampler):
    def __init__(self, worker_factory, agents, encoders, make_env):
        # pylint: disable=super-init-not-called
        self._factory = worker_factory
        self._agents = worker_factory.prepare_worker_messages(agents)
        self._encoders = worker_factory.prepare_worker_messages(encoders)
        self._envs = worker_factory.prepare_worker_messages(make_env, preprocess=copy.deepcopy)
        self._workers = [
            worker_factory(i) for i in range(worker_factory.n_workers)
        ]
        for worker, agent, encoder, env in zip(self._workers, self._agents, self._encoders, self._envs):
            worker.update_agent(agent)
            worker.update_encoder(encoder)
            worker.update_env(env())

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, encoders, make_env):
        """Construct this sampler.

        Args:
            worker_factory (WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents (Agent or List[Agent]): Agent(s) to use to perform rollouts.
                If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs (gym.Env or List[gym.Env]): Environment rollouts are performed
                in. If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        return cls(worker_factory, agents, encoders, make_env)

    def _update_workers(self, agent_update, encoder_update, env_update, worker_update):
        """Apply updates to the workers.

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        encoder_updates = self._factory.prepare_worker_messages(encoder_update)
        env_updates = self._factory.prepare_worker_messages(env_update, preprocess=copy.deepcopy)
        worker_updates = self._factory.prepare_worker_messages(worker_update)
        for worker, agent_up, encoder_up, env_up, worker_up in zip(self._workers, agent_updates,
                                            encoder_updates, env_updates, worker_updates):
            worker.update_agent(agent_up)
            worker.update_encoder(encoder_up)
            worker.update_env(env_up)
            worker.update_worker(worker_up)

    def obtain_exact_trajectories(self,
                                  n_traj_per_worker,
                                  agent_update,
                                  encoder_update,
                                  env_update=None,
                                  worker_update=None,
                                  get_attrs=None):
        self._update_workers(agent_update, encoder_update, env_update, worker_update)
        trajectories = []
        log_data_list = []
        for worker, n_traj in zip(self._workers, n_traj_per_worker):
            for _ in range(n_traj):
                batch, log_data = worker.rollout()
                trajectories.append(batch)

                if log_data:
                    log_data_list.append(log_data)

        if log_data_list:
            log_dict = process_log_data(log_data_list, trajectories)
        else:
            log_dict = {}

        infos = defaultdict(list)
        if get_attrs is not None:
            for i in range(self._factory.n_workers):
                contents = self._workers[i].get_attrs(get_attrs)
                for k, v in contents.items():
                    infos[k].append(v)

        return TrajectoryBatch.concatenate(*trajectories), infos, log_dict

def process_log_data(log_data_list, trajectories):
    r_squares = np.array([elem["r_square"] for elem in log_data_list])
    pearsons = np.array([elem["pearson"] for elem in log_data_list])
    returns = np.array([sum(elem.rewards) for elem in trajectories])
    returns_argmax = np.argmax(returns)
    returns_argmin = np.argmin(returns)
    r_square_objects = np.array(
        [elem.get("r_square_object", np.nan) for elem in log_data_list], dtype=float
    )
    pearson_objects = np.array(
        [elem.get("pearson_object", np.nan) for elem in log_data_list], dtype=float
    )

    log_dict = {
        # Record R^2 for phi(s)
        "r_square_min": np.min(r_squares),
        "r_square_mean": np.mean(r_squares),
        "r_square_max": np.max(r_squares),
        "r_square_std": np.std(r_squares),
        "r_square_for_max_return": r_squares[returns_argmax],
        "r_square_for_min_return": r_squares[returns_argmin],
        # Record Pearson for phi(s)
        "pearson_min": np.min(pearsons),
        "pearson_mean": np.mean(pearsons),
        "pearson_max": np.max(pearsons),
        "pearson_std": np.std(pearsons),
        "pearson_for_max_return": pearsons[returns_argmax],
        "pearson_for_min_return": pearsons[returns_argmin],
        # Record max and min return
        "max_return": np.max(returns),
        "min_return": np.min(returns),
    }

    r_square_object_valid = r_square_objects[~np.isnan(r_square_objects)]
    if len(r_square_object_valid) > 0:
        log_dict["r_square_object_min"] = np.min(r_square_object_valid)
        log_dict["r_square_object_mean"] = np.mean(r_square_object_valid)
        log_dict["r_square_object_max"] = np.max(r_square_object_valid)
        log_dict["r_square_object_std"] = np.std(r_square_object_valid)
    else:
        log_dict["r_square_object_min"] = np.nan
        log_dict["r_square_object_mean"] = np.nan
        log_dict["r_square_object_max"] = np.nan
        log_dict["r_square_object_std"] = np.nan

    pearson_object_valid = pearson_objects[~np.isnan(pearson_objects)]
    if len(pearson_object_valid) > 0:
        log_dict["pearson_object_min"] = np.min(pearson_object_valid)
        log_dict["pearson_object_mean"] = np.mean(pearson_object_valid)
        log_dict["pearson_object_max"] = np.max(pearson_object_valid)
        log_dict["pearson_object_std"] = np.std(pearson_object_valid)
    else:
        log_dict["pearson_object_min"] = np.nan
        log_dict["pearson_object_mean"] = np.nan
        log_dict["pearson_object_max"] = np.nan
        log_dict["pearson_object_std"] = np.nan

    log_dict["r_square_object_for_max_return"] = r_square_objects[returns_argmax]
    log_dict["r_square_object_for_min_return"] = r_square_objects[returns_argmin]
    log_dict["pearson_object_for_max_return"] = pearson_objects[returns_argmax]
    log_dict["pearson_object_for_min_return"] = pearson_objects[returns_argmin]

    # Process multi-step differences (including step 1)
    step_sizes = [1, 2, 3, 4, 5, 10, 20]
    for step_size in step_sizes:
        # R^2 for multi-step differences
        key_r2 = f"r_square_diff_{step_size}_step"
        key_pearson = f"pearson_diff_{step_size}_step"

        # Get values, using np.nan as default
        r2_multi = np.array([elem.get(key_r2, np.nan) for elem in log_data_list])
        pearson_multi = np.array([elem.get(key_pearson, np.nan) for elem in log_data_list])

        # Filter out NaN values for statistics
        r2_valid = r2_multi[~np.isnan(r2_multi)]
        pearson_valid = pearson_multi[~np.isnan(pearson_multi)]

        # Always add entries, use NaN when no valid data
        if len(r2_valid) > 0:
            log_dict[f"{key_r2}_min"] = np.min(r2_valid)
            log_dict[f"{key_r2}_mean"] = np.mean(r2_valid)
            log_dict[f"{key_r2}_max"] = np.max(r2_valid)
            log_dict[f"{key_r2}_std"] = np.std(r2_valid)
        else:
            log_dict[f"{key_r2}_min"] = np.nan
            log_dict[f"{key_r2}_mean"] = np.nan
            log_dict[f"{key_r2}_max"] = np.nan
            log_dict[f"{key_r2}_std"] = np.nan

        if len(pearson_valid) > 0:
            log_dict[f"{key_pearson}_min"] = np.min(pearson_valid)
            log_dict[f"{key_pearson}_mean"] = np.mean(pearson_valid)
            log_dict[f"{key_pearson}_max"] = np.max(pearson_valid)
            log_dict[f"{key_pearson}_std"] = np.std(pearson_valid)
        else:
            log_dict[f"{key_pearson}_min"] = np.nan
            log_dict[f"{key_pearson}_mean"] = np.nan
            log_dict[f"{key_pearson}_max"] = np.nan
            log_dict[f"{key_pearson}_std"] = np.nan

    for step_size in step_sizes:
        key_r2 = f"r_square_diff_{step_size}_step_object"
        key_pearson = f"pearson_diff_{step_size}_step_object"

        r2_multi = np.array([elem.get(key_r2, np.nan) for elem in log_data_list])
        pearson_multi = np.array([elem.get(key_pearson, np.nan) for elem in log_data_list])

        r2_valid = r2_multi[~np.isnan(r2_multi)]
        pearson_valid = pearson_multi[~np.isnan(pearson_multi)]

        if len(r2_valid) > 0:
            log_dict[f"{key_r2}_min"] = np.min(r2_valid)
            log_dict[f"{key_r2}_mean"] = np.mean(r2_valid)
            log_dict[f"{key_r2}_max"] = np.max(r2_valid)
            log_dict[f"{key_r2}_std"] = np.std(r2_valid)
        else:
            log_dict[f"{key_r2}_min"] = np.nan
            log_dict[f"{key_r2}_mean"] = np.nan
            log_dict[f"{key_r2}_max"] = np.nan
            log_dict[f"{key_r2}_std"] = np.nan

        if len(pearson_valid) > 0:
            log_dict[f"{key_pearson}_min"] = np.min(pearson_valid)
            log_dict[f"{key_pearson}_mean"] = np.mean(pearson_valid)
            log_dict[f"{key_pearson}_max"] = np.max(pearson_valid)
            log_dict[f"{key_pearson}_std"] = np.std(pearson_valid)
        else:
            log_dict[f"{key_pearson}_min"] = np.nan
            log_dict[f"{key_pearson}_mean"] = np.nan
            log_dict[f"{key_pearson}_max"] = np.nan
            log_dict[f"{key_pearson}_std"] = np.nan

    sac_payloads = [elem.get("_sac_eval_data") for elem in log_data_list if "_sac_eval_data" in elem]
    metra_states_list = [elem.get("_metra_states") for elem in log_data_list if "_metra_states" in elem]
    sac_metrics = aggregate_sac_metrics(sac_payloads, metra_states_list)
    log_dict.update(sac_metrics)

    return log_dict
