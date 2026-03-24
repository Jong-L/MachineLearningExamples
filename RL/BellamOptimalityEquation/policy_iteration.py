"""
由于整个示例中采用的环境比较简单，直接根据策略求解状态值对算力要求也不高，所以也给出策略迭代的示例
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld


@dataclass
class PolicyIterationConfig:
    threshold: float = 1e-6
    max_iter: int = 1000
    verbose: bool = True#是否打印迭代信息

@dataclass
class PolicyIterationResult:
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

def policy_iteration(env: GridWorld, cfg: PolicyIterationConfig):
    policy=np.ones((env.n_states, env.n_actions))/env.n_actions#初始化策略为均匀分布
    v=np.zeros(env.n_states, dtype=float)
    delta = np.inf
    converged = False
    iterations = 0

    for count in range(cfg.max_iter):
        #policy evaluation
        new_v=env.get_true_value_by_policy(policy)#与值迭代的区别在于这里是求解状态值

        #policy improvement
        for s_idx in range(env.n_states):
            q_vals = np.array([q_value(env, v, s_idx, action) for action in env.actions], dtype=float)#所有动作的q值
            best_a = int(np.argmax(q_vals))
            policy[s_idx] = np.eye(env.n_actions)[best_a]
        
        delta = np.max(np.abs(v - new_v))
        v = new_v
        iterations = count + 1
        if delta < cfg.threshold:
            converged = True
            break
            
    if cfg.verbose:
        if converged:
            print(f"Converged at iteration {iterations}, delta={delta:.3e}")
        else:
            print(f"Reached maximum iterations ({cfg.max_iter}), last delta={delta:.3e}")

    return PolicyIterationResult(
        value=v,
        policy=policy,
        iterations=iterations,
        delta=delta,
        converged=converged,
    )

if __name__ == "__main__":
    env = GridWorld()
    result=policy_iteration(env, PolicyIterationConfig())
    env = GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(result.value, title="Value Iteration V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
