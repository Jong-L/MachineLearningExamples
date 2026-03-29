"""
Q-Learning (off-policy version)
这个在赵世钰老师的课程中被称作 Q-Learning (off-policy version)
version1是动作策略在学习过程中也不断变好
version2是动作策略在学习过程中完全不更新

可以发现这个算法的结果和策略迭代或值迭代的结果一样，状态值误差为0
因为q learning本质上就是在求解贝尔曼最优公式

而MC和sarsa(以及version1)的问题在于使用了epsilon greedy策略来保证探索，虽然形式上是在求解贝尔曼最优公式，
但是每次更新q值时实际上是在估计q值的期望值，这一步是在求解贝尔曼期望方程，
其得到的最优动作永远有“保守”的成分，不是完完全全地利用(exploit)
而在理想的GLIE（Greedy-Like In Expectation）条件下，
MC和sarsa也能够得到和策略迭代或值迭代相同的结果，即状态值的误差趋于0
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
class QLearningConfig:
    alpha: float = 0.005
    n_episodes: int = 2000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100  # 连续 patience 次 q 值变化小于 threshold 则认为收敛

@dataclass
class QLearningResult:
    policy: np.ndarray  # 目标策略（greedy 策略）
    iterations: int
    converged: bool

def q_learning(env: GridWorld, config: QLearningConfig) -> QLearningResult:
    rng = np.random.default_rng(config.seed)
    behavior_policy = env.policy  # 行为策略 π_b
    target_policy = np.zeros((env.n_states, env.n_actions))  # 目标策略 π_T（greedy 策略），需要更新
    q_table = np.zeros((env.n_states, env.n_actions))
    last_q_table = np.zeros((env.n_states, env.n_actions))
    no_improvement_count = 0
    converged = False
    iterations = 0

    for count in range(config.n_episodes):
        state_idx = rng.choice(env.n_states)
        state = env.index_to_state(state_idx)  # s_t
        
        action_id = rng.choice(env.n_actions, p=behavior_policy[state_idx])
        action = env.actions[action_id]  # a_t

        for i in range(config.episode_length):
            state_next, reward = env.step(state, action)  # s_{t+1}, r_{t+1}
            state_idx = env.state_to_index(state)
            action_idx = env.actions.index(action)
            state_next_idx = env.state_to_index(state_next)
            
            max_q_next = np.max(q_table[state_next_idx])
            td_error = reward + env.gamma * max_q_next - q_table[state_idx, action_idx]
            q_table[state_idx, action_idx] += config.alpha * td_error

            best_action_idx = np.argmax(q_table[state_idx])
            target_policy[state_idx] = np.zeros(env.n_actions)
            target_policy[state_idx][best_action_idx] = 1.0
            
            action_next_id = rng.choice(env.n_actions, p=behavior_policy[state_next_idx])
            action = env.actions[action_next_id]
            state = state_next
        
        # 检查收敛性
        delta = np.max(np.abs(q_table - last_q_table))
        
        iterations += 1
        if delta < config.threshold:
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                converged = True
                break
        else:
            no_improvement_count = 0

        last_q_table = np.copy(q_table)

    # 返回目标策略
    return QLearningResult(policy=target_policy, iterations=iterations, converged=converged)

if __name__ == "__main__":
    env = GridWorld()
    cfg = QLearningConfig()
    start_time = time.perf_counter()
    result = q_learning(env, cfg)

    elapsed_time = time.perf_counter() - start_time

    # 可视化结果（使用目标策略）
    env = GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value = env.get_true_value_by_policy(result.policy)
    env.render_with_state_value(state_value, title="Q-Learning V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    print("\n=== 运行配置与耗时 ===")
    print("算法：Q-Learning (off-policy version)")
    print(f"配置参数：{asdict(cfg)}")
    print(f"迭代信息：iterations={result.iterations}, converged={result.converged}")
    print(f"算法运行时间：{elapsed_time:.6f} 秒")

    env_id = generate_env_id(env)
    if has_optimal_solution(env_id):
        optimal_solution = load_optimal_solution(env_id)
        true_state_value = optimal_solution.value
        result_value_sum = np.sum(state_value)
        true_value_sum = np.sum(true_state_value)
        error = np.mean(np.abs(true_state_value - state_value))

        print(f"总状态值：{result_value_sum:.5f}, 最优总状态值：{true_value_sum:.5f}")
        print(f"mean abs error: {error:.5f}")
