"""Sampler that runs workers in the main process."""
import copy
from collections import defaultdict
import numpy as np

from garage import TrajectoryBatch
from garage.sampler import LocalSampler


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
        batches = []
        r_squares = []
        for worker, n_traj in zip(self._workers, n_traj_per_worker):
            for _ in range(n_traj):
                batch, r_square = worker.rollout()
                batches.append(batch)

                if r_square is not None:
                    r_squares.append(r_square)

        if r_squares:
            r_squares = np.array(r_squares)
            r_square_dict = {
                "r_square_min": np.min(r_squares),
                "r_square_mean": np.mean(r_squares),
                "r_square_max": np.max(r_squares),
                "r_square_std": np.std(r_squares)
            }
        else:
            r_square_dict = {}

        infos = defaultdict(list)
        if get_attrs is not None:
            for i in range(self._factory.n_workers):
                contents = self._workers[i].get_attrs(get_attrs)
                for k, v in contents.items():
                    infos[k].append(v)

        return TrajectoryBatch.concatenate(*batches), infos, r_square_dict
