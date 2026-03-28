import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from dataclasses import asdict, dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld
from optimal_solution_manager import (
    generate_env_id,
    load_optimal_solution,
    has_optimal_solution
)

@dataclass
class NStepSARSAConfig:
    alpha: float = 0.005
    epsilon: float = 0.1
    n_episodes: int = 2000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    n_steps: int = 10#n-step SARSA中的n

@dataclass
class NStepSARSAResult:
    policy: np.ndarray
    q_table: np.ndarray
    converged: bool
    iterations: int

def n_step_sarsa(env: GridWorld, config: NStepSARSAConfig) -> NStepSARSAResult:
    rng=np.random.default_rng(config.seed)
    policy=env.policy
    q_table=np.zeros((env.n_states,env.n_actions))
    last_q_table=np.zeros((env.n_states,env.n_actions))
    no_improvement_count=0
    converged=False
    iterations=0

    for count in range(config.n_episodes):
        state_idx=rng.choice(env.n_states)
        state=env.index_to_state(state_idx)
        action_idx=rng.choice(env.n_actions, p=policy[state_idx])
        action=env.actions[action_idx]
