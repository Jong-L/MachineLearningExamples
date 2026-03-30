"""
与TD(0)相比，SARSA是在估计q值，并依据q值来更新策略。
如果连续多个episode q值没有得到明显改进则认为收敛
"""

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
class SARSAConfig:
    alpha: float = 0.005
    epsilon: float = 0.1#epsilon greedy policy
    n_episodes: int = 2000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100#连续patience次q值变化小于threshold则认为收敛

@dataclass
class SARSAResult:
    policy: np.ndarray
    iterations: int
    converged: bool

def sarsa(env: GridWorld, config: SARSAConfig) -> SARSAResult:
    rng=np.random.default_rng(config.seed)
    policy=env.policy#初始化策略
    q_table=np.zeros((env.n_states,env.n_actions))
    last_q_table=np.zeros((env.n_states,env.n_actions))
    no_improvement_count=0
    converged=False
    iterations=0

    for count in range(config.n_episodes):
        state_idx=rng.choice(env.n_states)
        state=env.index_to_state(state_idx)#s_t
        action_idx = rng.choice(env.n_actions, p=policy[state_idx])
        action = env.actions[action_idx]  # a_t

        for i in range(config.episode_length):
            state_next,reward=env.step(state,action)#s_{t+1},r_{t+1}
            state_idx=env.state_to_index(state)
            action_idx=env.actions.index(action)
            state_next_idx=env.state_to_index(state_next)

            # 根据 s_{t+1} 的当前策略采样 a_{t+1}
            action_next_idx = rng.choice(env.n_actions, p=policy[state_next_idx])
            action_next = env.actions[action_next_idx]

            q_table[state_idx,action_idx]+=config.alpha*(reward+env.gamma*q_table[state_next_idx,action_next_idx]-q_table[state_idx,action_idx])

            #更新策略，epsilon greedy
            best_action_idx=np.argmax(q_table[state_idx])
            policy[state_idx]=np.full((env.n_actions), config.epsilon/env.n_actions)
            policy[state_idx][best_action_idx]=1-config.epsilon+config.epsilon/env.n_actions

            action=action_next
            state=state_next
        
        delta=np.max(np.abs(q_table-last_q_table))

        iterations+=1
        if delta<config.threshold:
            no_improvement_count+=1
            if no_improvement_count>=config.patience:
                converged=True
                #break
                #增量学习更新量小，不设跳出。
        else:
            no_improvement_count=0

        last_q_table=np.copy(q_table)

    return SARSAResult(policy=policy,iterations=iterations,converged=converged)

if __name__ == "__main__":
    env = GridWorld()
    cfg = SARSAConfig()
    start_time = time.perf_counter()
    result = sarsa(env, cfg)
    elapsed_time = time.perf_counter() - start_time

    env=GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value=env.get_true_value_by_policy(result.policy)
    env.render_with_state_value(state_value, title="Sarsa V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
    #将soft policy 转为hard policy，重新查看啊结果
    best_action_idx=np.argmax(result.policy,axis=1)
    policy=np.array([np.eye(env.n_actions)[best_action_idx[s_idx]] for s_idx in range(env.n_states)])
    env=GridWorld(policy=policy)
    state_value=env.get_true_value_by_policy(policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(state_value, title="Sarsa Value ", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    print("\n=== 运行配置与耗时 ===")
    print("算法: sarsa")
    print(f"配置参数: {asdict(cfg)}")
    print(f"迭代信息: iterations={result.iterations}, converged={result.converged}")
    print(f"算法运行时间: {elapsed_time:.6f} 秒")

    env_id=generate_env_id(env)
    if has_optimal_solution(env_id):
        optimal_solution = load_optimal_solution(env_id)
        true_state_value = optimal_solution.value
        result_value_sum = np.sum(state_value)
        true_value_sum = np.sum(true_state_value)
        error = np.mean(np.abs(true_state_value - state_value))
        
        print(f"总状态值: {result_value_sum:.5f}, 最优总状态值: {true_value_sum:.5f}")
        print(f"mean abs error: {error:.5f}")

            
