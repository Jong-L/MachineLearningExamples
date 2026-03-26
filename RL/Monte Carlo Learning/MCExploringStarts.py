"""
Monte Carlo Exploring Starts (MCES) with first visit
相比于MCBasic，MCES在一个episode中同时计算多个q值，而不是一个episode只计算一个q值，
这样能够用更少的episode估计到q值，提高了样本利用率。
将起点选择改为随机，更均匀地访问每个状态，相比于依序遍历每个起点，
能在更少的迭代情况下访问到所有状态，能够降低计算成本。

这里的策略是确定性的，所以可能会出现某些状态一直没被访问，导致效果不好，
在下一节中会引入soft policy来改进
"""

from pstats import StatsProfile
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

@dataclass
class MCESConfig:
    threshold: float = 1e-6
    max_iter: int = 1000
    episode_length: int = 150
    seed: int = 42
    verbose: bool = True#是否打印迭代信息

@dataclass
class MCESResult:
    value: np.ndarray
    policy: np.ndarray
    iterations: int
    delta: float
    converged: bool

# 随机选择状态-动作
def sample_randomly(env: GridWorld) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    state = env.index_to_state(np.random.randint(env.n_states))
    action = env.actions[np.random.randint(env.n_actions)]
    return state, action

# 生成episode
def episode_generate(env: GridWorld, state: Tuple[int, int], action: Tuple[int, int],policy: np.ndarray,episode_length: int,seed:int) -> List:
    episode = []
    #rng=np.random.default_rng(seed)
    rng=np.random.default_rng()
    next_state, reward = env.step(state, action)
    episode.append((state, action, reward))
    state=next_state
    for _ in range(episode_length):
        next_state,action,reward=env.sample_next(state,policy,rng)
        episode.append((state, action, reward))
        state = next_state
    
    return episode


def mc_exploring_starts(env: GridWorld, config: MCESConfig):
    Returns = np.zeros((env.n_states, env.n_actions))
    policy=np.ones((env.n_states, env.n_actions))/env.n_actions#初始化策略
    v=env.get_true_value_by_policy(policy)
    converged = False
    for i in range(config.max_iter):#for each episode
        q_episode=np.zeros((env.n_states, env.n_actions))
        state, action = sample_randomly(env)
        episode = episode_generate(env, state, action, policy, config.episode_length, config.seed)
        g=0
        for state, action, reward in reversed(episode):
            g=reward+env.gamma*g
            q_episode[env.state_to_index(state), env.actions.index(action)] = g#first visit, overwrite
        
        Returns += q_episode
        q_value=Returns / (i + 1)
        #policy improvement
        best_action_idx = np.argmax(q_value, axis=1)
        policy=np.array([np.eye(env.n_actions)[best_action_idx[s_idx]] for s_idx in range(env.n_states)])
        
        new_v=env.get_true_value_by_policy(policy)
        delta = np.max(np.abs(new_v - v))
        if delta < config.threshold:
            converged = True
            #break  # MCES 每个 episode 只访问部分状态，很多 Q 值未被更新
            # 固定迭代次数能保证充分的探索，避免过早收敛到次优解因此强制迭代次数达到max_iter

        v=new_v

    if config.verbose:
        if converged:
            print(f"MCES converged at iteration {i + 1} with delta {delta}")
        else:
            print(f"Reached maximum iterations ({cfg.max_iter}), last delta={delta:.3e}")

    state_value=env.get_true_value_by_policy(policy)
    return MCESResult(
        value=state_value,
        policy=policy,
        iterations=i + 1,
        delta=delta,
        converged=converged,
    )

if __name__ == "__main__":
    env = GridWorld()
    config = MCESConfig()
    result=mc_exploring_starts(env, config)
    env=GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(result.value, title="MCES Value ", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
