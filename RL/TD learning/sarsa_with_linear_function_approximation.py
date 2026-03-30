"""
SARSA with Linear Function Approximation
在 SARSA 算法中使用线性函数近似 Q 值
特征向量构造方法与 TD-Linear_state_value.py 一致
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
class SarsaLinearConfig:
    alpha: float = 0.005
    epsilon: float = 0.1  # epsilon greedy policy
    n_episodes: int = 2000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100  # 连续 patience 次参数变化小于 threshold 则认为收敛

@dataclass
class SarsaLinearResult:
    theta: np.ndarray  # 线性函数参数
    iterations: int
    converged: bool


def build_features(env):
    """
    phi(s,a) = [phi(s,a_1), phi(s,a_2), ..., phi(s,a_env.n_actions)]^T
    其中 phi_i(s,a) 只在第 i 个维度有值，其余为 0
    """
    features = np.zeros((env.n_states, env.n_actions, 6))
    
    for s in range(env.n_states):
        r, c = env.index_to_state(s)
        rn = r / (env.rows - 1)  # 归一化的行坐标
        cn = c / (env.cols - 1)  # 归一化的列坐标
        
        state_feature = np.array([
            1.0,
            rn,
            cn,
            rn * cn,
            rn ** 2,
            cn ** 2,
        ])
        
        for a in range(env.n_actions):
            features[s, a] = state_feature
    
    return features

def q_value(theta, features, state_idx, action_idx):
    return features[state_idx, action_idx] @ theta

def get_greedy_action(theta, features, state_idx, env):
    """根据当前 Q 值函数选择贪婪动作"""
    q_values = np.array([q_value(theta, features, state_idx, a) for a in range(env.n_actions)])
    return np.argmax(q_values)

def sarsa_linear(env: GridWorld, config: SarsaLinearConfig) -> SarsaLinearResult:
    rng = np.random.default_rng(config.seed)
    features = build_features(env)
    theta = np.zeros(features.shape[2])  # 初始化参数向量为零向量
    last_theta = np.copy(theta)
    no_improvement_count = 0
    converged = False
    iterations = 0
    
    for count in range(config.n_episodes):
        state_idx = rng.choice(env.n_states)
        state = env.index_to_state(state_idx)  # s_t
        
        # epsilon-greedy 策略选择动作
        if rng.random() < config.epsilon:
            action_idx = rng.choice(env.n_actions)
        else:
            action_idx = get_greedy_action(theta, features, state_idx, env)
        action = env.actions[action_idx]  # a_t
        
        for i in range(config.episode_length):
            state_next, reward = env.step(state, action)  # s_{t+1}, r_{t+1}
            state_next_idx = env.state_to_index(state_next)
            
            # 根据 s_{t+1} 的当前策略采样 a_{t+1}
            if rng.random() < config.epsilon:
                action_next_idx = rng.choice(env.n_actions)
            else:
                action_next_idx = get_greedy_action(theta, features, state_next_idx, env)
            action_next = env.actions[action_next_idx]
            
            # 计算 TD 误差
            q_s = q_value(theta, features, state_idx, action_idx)
            q_next = q_value(theta, features, state_next_idx, action_next_idx)
            delta = reward + env.gamma * q_next - q_s
            
            # 更新参数
            theta += config.alpha * delta * features[state_idx, action_idx]
            
            # 更新策略
            action = action_next
            state = state_next
            state_idx = state_next_idx
            action_idx = action_next_idx
        
        # 检查收敛性
        delta_theta = np.max(np.abs(theta - last_theta))
        iterations += 1
        
        if delta_theta < config.threshold:
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                converged = True
        else:
            no_improvement_count = 0
        
        last_theta = np.copy(theta)
    
    return SarsaLinearResult(theta=theta, iterations=iterations, converged=converged)


def extract_policy(theta, features, env):
    """从学习到的 Q 函数中提取确定性策略"""
    policy = np.zeros((env.n_states, env.n_actions))
    
    for s in range(env.n_states):
        q_values = np.array([q_value(theta, features, s, a) for a in range(env.n_actions)])
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy


if __name__ == "__main__":
    env = GridWorld()
    cfg = SarsaLinearConfig(
        alpha=0.005,
        epsilon=0.1,
        n_episodes=2000,
        episode_length=100,
        seed=42,
        threshold=1e-6,
        patience=100
    )
    
    start_time = time.perf_counter()
    result = sarsa_linear(env, cfg)
    elapsed_time = time.perf_counter() - start_time
    
    # 提取策略并可视化
    features = build_features(env)
    policy = extract_policy(result.theta, features, env)
    env_with_policy = GridWorld(policy=policy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value = env_with_policy.get_true_value_by_policy(policy)
    env_with_policy.render_with_state_value(state_value, title="SARSA Linear V*(s)", ax=ax1)
    env_with_policy.render_with_policy(ax=ax2)
    plt.show()
    
    print("\n=== 运行配置与耗时 ===")
    print("算法: SARSA with Linear Function Approximation")
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
