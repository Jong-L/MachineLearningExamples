"""
将策略迭代中q值的计算方法改为monte carlo方法
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
from dataclasses import asdict, dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld
from optimal_solution_manager import (
    generate_env_id,
    load_optimal_solution,
    has_optimal_solution
)

@dataclass
class PolicyIterationConfig:
    threshold: float = 1e-6
    max_iter: int = 500#策略迭代数
    episode_length: int = 15#估计q值的episode长度
    n_episodes: int = 5#估计q值的episode数
    seed:int=42

@dataclass
class PolicyIterationResult:
    value: np.ndarray
    policy: np.ndarray
    iterations: int
    delta: float
    converged: bool

def monte_carlo_q_value(env: GridWorld, state_idx: int, action: 
                        Tuple[int, int],policy: np.ndarray,
                        episodes:int,episode_length: int,seed:int):
    rng=np.random.RandomState(seed)#固定随机种子
    gamma=env.gamma
    #gamma=1
    start_state=env.index_to_state(state_idx)
    q=np.zeros(episodes)#q[i]表示第i个episode的q值
    for i in range(episodes):
        state,reward=env.step(start_state,action)
        q[i]+=reward
        for t in range(episode_length):
            next_state,action, reward = env.sample_next(state,policy, rng)
            q[i]+=reward*(gamma**(t+1))
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
            q_vals = np.array([monte_carlo_q_value(env, s_idx, action,policy,cfg.n_episodes,cfg.episode_length,cfg.seed) 
                               for action in env.actions])#所有动作的q值
            best_a = np.argmax(q_vals)
            
            policy[s_idx] = np.eye(env.n_actions)[best_a]
        
        delta = np.max(np.abs(v - new_v))
        v = np.copy(new_v)
        iterations = count + 1
        if delta < cfg.threshold:
            converged = True
            break

    return PolicyIterationResult(
        value=v,
        policy=policy,
        iterations=iterations,
        delta=delta,
        converged=converged,
    )

if __name__ == "__main__":
    env = GridWorld()
    cfg = PolicyIterationConfig()
    start_time = time.perf_counter()
    result=policy_iteration(env, cfg)

    elapsed_time = time.perf_counter() - start_time

    env = GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(result.value, title="MC Value ", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
    
    print("\n=== 运行配置与耗时 ===")
    print("算法: MC-based Policy Iteration (MCBasic)")
    print(f"配置参数: {asdict(cfg)}")
    print(f"迭代信息: iterations={result.iterations}, converged={result.converged}, delta={result.delta:.3e}")
    print(f"算法运行时间: {elapsed_time:.6f} 秒")
    env_id=generate_env_id(env)
    if has_optimal_solution(env_id):
        optimal_solution = load_optimal_solution(env_id)
        true_state_value = optimal_solution.value
        result_value_sum = np.sum(result.value)
        true_value_sum = np.sum(true_state_value)
        error = np.mean(np.abs(true_state_value - result.value))
        
        print(f"总状态值: {result_value_sum:.5f}, 最优总状态值: {true_value_sum:.5f}")
        print(f"mean abs error: {error:.5f}")
