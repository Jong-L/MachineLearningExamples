"""
SARSA with GLIE (Greedy in the Limit with Infinite Exploration)

GLIE 条件：
1. 无限次访问所有状态 - 动作对（Infinite Exploration）
2. 策略最终贪婪化（Greedy in the Limit）

在 GLIE 条件下，SARSA 保证收敛到最优策略
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
class SARSAGLIEConfig:
    alpha: float = 0.005#如果学习率也满足RM算法的条件，结果应该更优
    epsilon_start: float = 1.0  # 初始 epsilon（充分探索）
    epsilon_end: float = 0.0    # 最终 epsilon（贪婪）
    epsilon_decay: float = 0.9999  # epsilon 衰减率
    n_episodes: int = 10000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100  # 连续 patience 次 q 值变化小于 threshold 则认为收敛

@dataclass
class SARSAGLIEResult:
    policy: np.ndarray
    q_table: np.ndarray
    iterations: int
    converged: bool
    epsilon_history: list  # 记录 epsilon 变化

def sarsa_glie(env: GridWorld, config: SARSAGLIEConfig) -> SARSAGLIEResult:
    """
    SARSA with GLIE 条件
    
    关键改进：
    1. epsilon 从 1.0 开始（充分探索）
    2. epsilon 指数衰减到 0（最终贪婪）
    3. 保证每个状态 - 动作对被无限次访问
    """
    rng = np.random.default_rng(config.seed)
    policy = env.policy  # 初始化策略
    q_table = np.zeros((env.n_states, env.n_actions))
    last_q_table = np.zeros((env.n_states, env.n_actions))
    no_improvement_count = 0
    converged = False
    iterations = 0
    current_epsilon = config.epsilon_start
    epsilon_history = []
    
    for count in range(config.n_episodes):
        # 记录当前 epsilon
        epsilon_history.append(current_epsilon)
        
        state_idx = rng.choice(env.n_states)
        state = env.index_to_state(state_idx)  # s_t
        
        # 使用当前 epsilon 选择动作
        action_id = rng.choice(env.n_actions, p=policy[state_idx])
        action = env.actions[action_id]  # a_t
        
        for i in range(config.episode_length):
            state_next, reward = env.step(state, action)  # s_{t+1}, r_{t+1}
            state_idx = env.state_to_index(state)
            action_idx = env.actions.index(action)
            state_next_idx = env.state_to_index(state_next)
            
            # 根据 s_{t+1} 的当前策略采样 a_{t+1}
            action_next_idx = rng.choice(env.n_actions, p=policy[state_next_idx])
            action_next = env.actions[action_next_idx]
            
            # SARSA 更新（贝尔曼期望方程）
            q_table[state_idx, action_idx] += config.alpha * (
                reward + env.gamma * q_table[state_next_idx, action_next_idx] - 
                q_table[state_idx, action_idx]
            )
            
            # 更新策略（epsilon-greedy）
            best_action_idx = np.argmax(q_table[state_idx])
            policy[state_idx] = np.full((env.n_actions), current_epsilon / env.n_actions)
            policy[state_idx][best_action_idx] = 1 - current_epsilon + current_epsilon / env.n_actions
            
            action = action_next
            state = state_next
        
        # GLIE 核心：epsilon 衰减
        current_epsilon = max(
            config.epsilon_end, 
            current_epsilon * config.epsilon_decay
        )
        
        # 检查收敛性
        delta = np.max(np.abs(q_table - last_q_table))
        
        if delta < config.threshold:
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                converged = True
                iterations = count
                #break
        else:
            no_improvement_count = 0
        
        last_q_table = np.copy(q_table)
        iterations = count
    
    print("算法结束时的epsilon:", current_epsilon)
    return SARSAGLIEResult(
        policy=policy,
        q_table=q_table,
        iterations=iterations,
        converged=converged,
        epsilon_history=epsilon_history
    )

if __name__ == "__main__":
    env = GridWorld()
    cfg = SARSAGLIEConfig()
    start_time = time.perf_counter()
    result = sarsa_glie(env, cfg)
    elapsed_time = time.perf_counter() - start_time
    
    # 将 soft policy 转为 hard policy，查看结果
    best_action_idx = np.argmax(result.policy, axis=1)
    deterministic_policy = np.array([
        np.eye(env.n_actions)[best_action_idx[s_idx]] 
        for s_idx in range(env.n_states)
    ])
    
    env = GridWorld(policy=deterministic_policy)
    state_value = env.get_true_value_by_policy(deterministic_policy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(state_value, title="SARSA-GLIE V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()
    
    print("\n=== 运行配置与耗时 ===")
    print("算法：SARSA with GLIE")
    print(f"配置参数：{asdict(cfg)}")
    print(f"迭代信息：iterations={result.iterations}, converged={result.converged}")
    print(f"算法运行时间：{elapsed_time:.6f} 秒")

    # 与最优解比较
    env_id = generate_env_id(env)
    if has_optimal_solution(env_id):
        optimal_solution = load_optimal_solution(env_id)
        true_state_value = optimal_solution.value
        result_value_sum = np.sum(state_value)
        true_value_sum = np.sum(true_state_value)
        error = np.mean(np.abs(true_state_value - state_value))
        
        # 与 Q-Learning 比较
        print(f"\n=== 对比分析 ===")
        print(f"SARSA-GLIE 误差：{error:.5f}")
        print(f"理论上 SARSA-GLIE 应收敛到最优策略（误差接近 0）")
        print(f"实际误差来源：有限 episode、衰减速度、探索充分性")
