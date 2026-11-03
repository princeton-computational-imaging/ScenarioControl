# Adapted from Scenario Dreamer (https://github.com/princeton-computational-imaging/scenario-dreamer).
# Changes: IDM planner only, several seeds per scenario, per-frame rollout saving and task splitting.
import os
import random

import hydra
import numpy as np
import torch
from tqdm import tqdm

from cfgs.config import CONFIG_PATH
from policies.idm_policy import IDMPolicy
from simulator import Simulator
from utils.sequence_saver import SequenceSaver
from utils.viz import generate_video


class PolicyEvaluator:
    """ Evaluate a given policy in a simulation environment over multiple scenarios."""
    def __init__(self, cfg, policy, env):
        """ Initialize the PolicyEvaluator."""
        self.cfg = cfg
        # policy being evaluated
        self.policy = policy
        # simulation environment
        self.env = env

    def reset(self):
        """ Reset the evaluator's statistics and random seeds."""
        self._seed_everything(self.cfg.seed)

        self.collision = []
        self.off_route = []
        self.completed = []
        self.progress = []

    @staticmethod
    def _seed_everything(seed):
        """ Re-seed torch, numpy and python random with the given seed."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def update_running_statistics(self, info):
        """ Update running statistics with info from the latest episode."""
        self.collision.append(info['collision'])
        self.off_route.append(info['off_route'])
        self.completed.append(info['completed'])
        self.progress.append(info['progress'])


    def compute_metrics(self):
        """ Compute evaluation metrics based on accumulated statistics."""
        metrics_dict = {
            'collision rate': np.array(self.collision).astype(float).mean(),
            'off route rate': np.array(self.off_route).astype(float).mean(),
            'completed rate': np.array(self.completed).astype(float).mean(),
            'progress': np.array(self.progress).astype(float).mean()
        }

        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]


    def evaluate_policy(self, scenario_idxs):
        """ Evaluate the policy on the given scenario indices into env.test_files.

        Each scenario is simulated with cfg.num_seeds seeds (seed, seed+1, ...); outputs are
        named <scenario>_seed<k>.
        """
        self.reset()

        for i in tqdm(scenario_idxs):
            sequence_id = os.path.splitext(os.path.basename(self.env.test_files[i]))[0]

            for seed in range(self.cfg.seed, self.cfg.seed + self.cfg.num_seeds):
                run_name = f'{sequence_id}_seed{seed}'
                print(f"Simulating environment {i} - {run_name}")
                # Re-seed so each run is independent and reproducible
                self._seed_everything(seed)

                obs = self.env.reset(i)
                self.policy.reset(obs)

                sequence_saver = None
                if self.cfg.save_sequence:
                    # reload the initial scene (env.scenario_dict is modified in reset)
                    sequence_saver = SequenceSaver(self.cfg.output_dir, run_name, self.env.load_initial_scene(i))
                    # frame 0 is the initial state, frame t+1 is the state after step t
                    sequence_saver.save_frame(0, self.env.get_serializable_state())

                terminated_info = None
                for t in range(self.env.steps):
                    if self.cfg.visualize and (not self.cfg.lightweight or t % 3 == 0):
                        self.env.render_state(name=run_name, movie_path=self.cfg.movie_path)

                    action = self.policy.act(obs)
                    obs, terminated, info = self.env.step(action)

                    if sequence_saver is not None:
                        sequence_saver.save_frame(t + 1, self.env.get_serializable_state())

                    if terminated and terminated_info is None:
                        terminated_info = info
                    if terminated and self.cfg.stop_on_termination:
                        break

                self.update_running_statistics(terminated_info)

                if self.cfg.visualize:
                    generate_video(name=run_name, output_dir=self.cfg.movie_path, delete_images=True)

                if self.cfg.verbose:
                    if self.cfg.behaviour_model.compute_metrics:
                        print("behaviour model metrics: ", self.env.behaviour_model.compute_metrics()[-1])
                    # policy metrics
                    print(self.compute_metrics()[-1])

        return self.compute_metrics()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    torch.manual_seed(cfg.sim.seed)
    random.seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    env = Simulator(cfg)
    assert env.num_test_scenarios > 0, f"No .pkl scenarios found in sim.dataset_path={cfg.sim.dataset_path}"
    assert cfg.sim.steps >= env.behaviour_model.cfg_dataset.train_context_length, "sim.steps must be >= the CtRL-Sim context length"

    scenario_idxs = list(range(env.num_test_scenarios))
    # with sim.task_id set (e.g. a Slurm array task), only process every num_tasks'th scenario
    if cfg.sim.task_id is not None:
        scenario_idxs = scenario_idxs[cfg.sim.task_id::cfg.sim.num_tasks]
        print(f"task {cfg.sim.task_id}/{cfg.sim.num_tasks}: {len(scenario_idxs)}/{env.num_test_scenarios} scenarios")

    policy = IDMPolicy(cfg, env)
    evaluator = PolicyEvaluator(cfg.sim, policy, env)
    _, metrics_str = evaluator.evaluate_policy(scenario_idxs)
    print(metrics_str)

if __name__ == "__main__":
    main()
