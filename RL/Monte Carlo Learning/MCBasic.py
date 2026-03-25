"""
将策略迭代中q值的计算方法改为monte carlo方法
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld


@dataclass
class PolicyIterationConfig:
    threshold: float = 1e-6
    max_iter: int = 1000#策略迭代数
    episode_length: int = 25#估计q值的episode长度
    episodes: int = 10#估计q值的episode数
    verbose: bool = True#是否打印迭代信息

@dataclass
class PolicyIterationResult:
    value: np.ndarray
    policy: np.ndarray
    iterations: int
    delta: float
    converged: bool

def monte_carlo_q_value(env: GridWorld, state_idx: int, action: 
                        Tuple[int, int],policy: np.ndarray,
                        episodes:int,episode_length: int):
    rng=np.random.RandomState(42)#固定随机种子
    gamma=env.gamma
    #gamma=1
    start_state=env.index_to_state(state_idx)
    q=np.zeros(episodes)#q[i]表示第i个episode的q值
    for i in range(episodes):
        state,reward=env.step(start_state,action)
        q[i]+=reward
        for t in range(episode_length):
            next_state, reward = env.sample_next(state,policy, rng)
            q[i]+=reward*(gamma**t)
            state=next_state
    q_value=np.mean(q)
    return q_value

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
            q_vals = np.array([monte_carlo_q_value(env, s_idx, action,policy,cfg.episodes,cfg.episode_length) 
                               for action in env.actions])#所有动作的q值
            best_a = np.argmax(q_vals)
            
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
    env.render_with_state_value(result.value, title="MC Value ", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
