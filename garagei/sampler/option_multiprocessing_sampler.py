"""A multiprocessing sampler which avoids waiting as much as possible."""

import itertools
import numpy as np
import torch.multiprocessing as mp
import multiprocessing.dummy as mpd
from collections import defaultdict

import click
import cloudpickle
import matplotlib
import setproctitle
from garage import TrajectoryBatch
from garage.sampler import MultiprocessingSampler


DEBUG = False
# DEBUG = True

if DEBUG:
    matplotlib.use("Agg")


class OptionMultiprocessingSampler(MultiprocessingSampler):
    def __init__(self, worker_factory, agents, encoders, make_env, n_thread):
        # pylint: disable=super-init-not-called
        self._factory = worker_factory
        self._agents = self._factory.prepare_worker_messages(agents, cloudpickle.dumps)
        self._encoders = self._factory.prepare_worker_messages(
            encoders, cloudpickle.dumps
        )
        self._envs = self._factory.prepare_worker_messages(make_env)
        self._n_thread = n_thread

        if not DEBUG:
            self._to_sampler = mp.Queue()
            self._to_worker = [mp.Queue() for _ in range(self._factory.n_workers)]
        else:
            self._to_sampler = mpd.Queue()
            self._to_worker = [mpd.Queue() for _ in range(self._factory.n_workers)]

        if not DEBUG:
            # If we crash from an exception, with full queues, we would rather not
            # hang forever, so we would like the process to close without flushing
            # the queues.
            # That's what cancel_join_thread does.
            for q in self._to_worker:
                q.cancel_join_thread()

        if not DEBUG:
            self._workers = [
                mp.Process(
                    target=run_worker,
                    kwargs=dict(
                        factory=self._factory,
                        to_sampler=self._to_sampler,
                        to_worker=self._to_worker[worker_number],
                        worker_number=worker_number,
                        agent=self._agents[worker_number],
                        encoder=self._encoders[worker_number],
                        env=self._envs[worker_number],
                        n_thread=self._n_thread,
                    ),
                    daemon=False,
                )
                for worker_number in range(self._factory.n_workers)
            ]
        else:
            self._workers = [
                mpd.Process(
                    target=run_worker,
                    kwargs=dict(
                        factory=self._factory,
                        to_sampler=self._to_sampler,
                        to_worker=self._to_worker[worker_number],
                        worker_number=worker_number,
                        agent=self._agents[worker_number],
                        encoder=self._encoders[worker_number],
                        env=self._envs[worker_number],
                        n_thread=self._n_thread,
                    ),
                )
                for worker_number in range(self._factory.n_workers)
            ]

        self._agent_version = 0
        self._encoder_version = 0
        for w in self._workers:
            w.start()

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, encoders, make_env, **kwargs):
        return cls(worker_factory, agents, encoders, make_env, **kwargs)

    def obtain_exact_trajectories(
        self,
        n_traj_per_workers,
        agent_update,
        encoder_update,
        env_update=None,
        worker_update=None,
        get_attrs=None,
    ):
        """Same as the parent method except that n_traj_per_workers can be either an integer or a list."""
        if isinstance(n_traj_per_workers, int):
            n_traj_per_workers = [n_traj_per_workers] * self._factory.n_workers

        self._agent_version += 1
        self._encoder_version += 1
        updated_workers = set()
        agent_ups = self._factory.prepare_worker_messages(
            agent_update, cloudpickle.dumps
        )
        encoder_ups = self._factory.prepare_worker_messages(
            encoder_update, cloudpickle.dumps
        )
        env_ups = self._factory.prepare_worker_messages(env_update)
        worker_ups = self._factory.prepare_worker_messages(worker_update)

        trajectories = defaultdict(list)
        log_data_list = defaultdict(list)
        for worker_number, q in enumerate(self._to_worker):
            q.put_nowait(
                (
                    "start",
                    (
                        agent_ups[worker_number],
                        encoder_ups[worker_number],
                        env_ups[worker_number],
                        worker_ups[worker_number],
                        self._agent_version,
                        self._encoder_version,
                    ),
                )
            )
            updated_workers.add(worker_number)
            if len(trajectories[worker_number]) < n_traj_per_workers[worker_number]:
                q.put_nowait(("rollout", ()))

        with click.progressbar(
            length=sum(n_traj_per_workers), label="Sampling"
        ) as pbar:
            while any(
                len(trajectories[i]) < n_traj_per_workers[i]
                for i in range(self._factory.n_workers)
            ):
                tag, contents = self._to_sampler.get()

                if tag == "trajectory":
                    pbar.update(1)
                    batch, log_data, agent_version, encoder_version, worker_n = contents

                    if (
                        agent_version == self._agent_version
                        and encoder_version == self._encoder_version
                    ):
                        trajectories[worker_n].append(batch)

                        if log_data:
                            log_data_list[worker_n].append(log_data)

                        if len(trajectories[worker_n]) < n_traj_per_workers[worker_n]:
                            self._to_worker[worker_n].put_nowait(("rollout", ()))
                        elif (
                            len(trajectories[worker_n]) == n_traj_per_workers[worker_n]
                        ):
                            self._to_worker[worker_n].put_nowait(("stop", ()))
                        else:
                            raise Exception(
                                "len(trajectories[worker_n]) > n_traj_per_workers[worker_n]"
                            )
                    else:
                        raise Exception("version mismatch")
                else:
                    raise AssertionError(
                        "Unknown tag {} with contents {}".format(tag, contents)
                    )

        ordered_trajectories = list(
            itertools.chain(*[trajectories[i] for i in range(self._factory.n_workers)])
        )
        ordered_log_data_list = list(
            itertools.chain(*[log_data_list[i] for i in range(self._factory.n_workers)])
        )

        if log_data_list:
            log_dict = process_log_data(ordered_log_data_list, ordered_trajectories)
        else:
            log_dict = {}

        infos = defaultdict(list)
        if get_attrs is not None:
            for i in range(self._factory.n_workers):
                self._to_worker[i].put_nowait(("get_attrs", get_attrs))
                tag, contents = self._to_sampler.get()
                assert tag == "attr_dict"
                for k, v in contents.items():
                    infos[k].append(v)

        return TrajectoryBatch.concatenate(*ordered_trajectories), infos, log_dict


def run_worker(
    factory, to_worker, to_sampler, worker_number, agent, encoder, env, n_thread
):
    if n_thread is not None:
        import torch

        torch.set_num_threads(n_thread)

    if not DEBUG:
        to_sampler.cancel_join_thread()
    setproctitle.setproctitle("worker:" + setproctitle.getproctitle())

    inner_worker = factory(worker_number)
    inner_worker.update_agent(cloudpickle.loads(agent))
    inner_worker.update_encoder(cloudpickle.loads(encoder))
    inner_worker.update_env(env())

    agent_version = 0
    encoder_version = 0

    while True:
        tag, contents = to_worker.get()

        if tag == "start":
            # Update env and policy.
            agent_update, encoder_update, env_update, worker_update, agent_version, encoder_version = contents
            inner_worker.update_agent(cloudpickle.loads(agent_update))
            inner_worker.update_encoder(cloudpickle.loads(encoder_update))
            inner_worker.update_env(env_update)
            inner_worker.update_worker(worker_update)
        elif tag == "stop":
            pass
        elif tag == "rollout":
            batch, log_data = inner_worker.rollout()
            to_sampler.put_nowait(("trajectory", (batch, log_data, agent_version, encoder_version, worker_number)))
        elif tag == "get_attrs":
            keys = contents
            attr_dict = inner_worker.get_attrs(keys)
            to_sampler.put_nowait(("attr_dict", attr_dict))
        elif tag == "exit":
            to_worker.close()
            to_sampler.close()
            inner_worker.shutdown()
            return
        else:
            raise AssertionError(
                "Unknown tag {} with contents {}".format(tag, contents)
            )

def process_log_data(log_data_list, trajectories):
    r_squares = np.array([elem["r_square"] for elem in log_data_list])
    r_square_diffs = np.array([elem["r_square_diff"] for elem in log_data_list])
    pearsons = np.array([elem["pearson"] for elem in log_data_list])
    pearson_diffs = np.array([elem["pearson_diff"] for elem in log_data_list])
    returns = np.array([sum(elem.rewards) for elem in trajectories])
    returns_argmax = np.argmax(returns)
    returns_argmin = np.argmin(returns)

    log_dict = {
        # Record R^2 for phi(s)
        "r_square_min": np.min(r_squares),
        "r_square_mean": np.mean(r_squares),
        "r_square_max": np.max(r_squares),
        "r_square_std": np.std(r_squares),
        "r_square_for_max_return": r_squares[returns_argmax],
        "r_square_for_min_return": r_squares[returns_argmin],
        # Record R^2 for phi(s) - phi(s')
        "r_square_diff_min": np.min(r_square_diffs),
        "r_square_diff_mean": np.mean(r_square_diffs),
        "r_square_diff_max": np.max(r_square_diffs),
        "r_square_diff_std": np.std(r_square_diffs),
        "r_square_diff_for_max_return": r_square_diffs[returns_argmax],
        "r_square_diff_for_min_return": r_square_diffs[returns_argmin],
        # Record Pearson for phi(s)
        "pearson_min": np.min(pearsons),
        "pearson_mean": np.mean(pearsons),
        "pearson_max": np.max(pearsons),
        "pearson_std": np.std(pearsons),
        "pearson_for_max_return": pearsons[returns_argmax],
        "pearson_for_min_return": pearsons[returns_argmin],
        # Record Pearson for phi(s) - phi(s')
        "pearson_diff_min": np.min(pearson_diffs),
        "pearson_diff_mean": np.mean(pearson_diffs),
        "pearson_diff_max": np.max(pearson_diffs),
        "pearson_diff_std": np.std(pearson_diffs),
        "pearson_diff_for_max_return": pearson_diffs[returns_argmax],
        "pearson_diff_for_min_return": pearson_diffs[returns_argmin],
        # Record max and min return
        "max_return": np.max(returns),
        "min_return": np.min(returns),
    }

    return log_dict
