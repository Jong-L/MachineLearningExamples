import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld
from optimal_solution_manager import (
    generate_env_id,
    save_optimal_solution,
    OptimalSolution
)

@dataclass
class ValueIterationConfig:
    threshold: float = 1e-6
    max_iter: int = 1000
    verbose: bool = True#是否打印迭代信息

@dataclass
class ValueIterationResult:
    value: np.ndarray
    policy: np.ndarray
    iterations: int
    delta: float
    converged: bool


def q_value(env: GridWorld, v: np.ndarray, state_idx: int, action: Tuple[int, int]):
    state = env.index_to_state(state_idx)
    next_state, reward = env.step(state, action)
    next_state_idx = env.state_to_index(next_state)
    return reward + env.gamma * v[next_state_idx]


def extract_policy(env: GridWorld, best_actions: np.ndarray) -> np.ndarray:
    policy = np.zeros((env.n_states, env.n_actions), dtype=float)
    policy[np.arange(env.n_states), best_actions] = 1.0
    return policy


def run_value_iteration(env: GridWorld, config: Optional[ValueIterationConfig] = None) -> ValueIterationResult:
    cfg = config or ValueIterationConfig()
    v = np.zeros(env.n_states, dtype=float)
    best_actions = np.zeros(env.n_states, dtype=int)
    delta = np.inf
    converged = False
    iterations = 0

    for count in range(cfg.max_iter):
        new_v = np.zeros(env.n_states, dtype=float)
        best_actions.fill(0)

        for s_idx in range(env.n_states):
            q_vals = np.array([q_value(env, v, s_idx, action) for action in env.actions])#所有动作的q值，注意传入的状态值是上一次迭代的值
            #policy update
            best_actions[s_idx] = np.argmax(q_vals)
            #value update
            new_v[s_idx] = np.max(q_vals)
        delta = np.max(np.abs(v - new_v))
        v = new_v
        iterations = count + 1
        if delta < cfg.threshold:
            converged = True
            break

    return ValueIterationResult(
        value=v,
        policy=extract_policy(env, best_actions),
        iterations=iterations,
        delta=delta,
        converged=converged,
    )

if __name__ == "__main__":
    env = GridWorld()
    cfg = ValueIterationConfig(threshold=1e-6, max_iter=1000, verbose=True)
    start_time = time.perf_counter()
    result = run_value_iteration(env, cfg)
    
    elapsed_time = time.perf_counter() - start_time
    print("\n=== 运行配置与耗时 ===")
    print("算法: Value Iteration")
    print(f"配置参数: {asdict(cfg)}")
    print(f"迭代信息: iterations={result.iterations}, converged={result.converged}, delta={result.delta:.3e}")
    print(f"算法运行时间: {elapsed_time:.6f} 秒")

    env = GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(result.value, title="Value Iteration V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()