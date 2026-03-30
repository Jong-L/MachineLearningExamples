"""
SARSA with Linear Function Approximation
在 SARSA 算法中使用线性函数近似 Q 值
特征向量构造方法与 TD-Linear_state_value.py 一致
可以看到效果很差，但逻辑上是正确的。
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
    n_episodes: int = 1000
    episode_length: int = 100
    seed: int = 42

@dataclass
class SarsaLinearResult:
    theta: np.ndarray  # 线性函数参数
    policy: np.ndarray
    iterations: int


def build_features(env):
    features = np.zeros((env.n_states, env.n_actions, 18))
    
    for s in range(env.n_states):
        r, c = env.index_to_state(s)
        rn = r / (env.rows - 1)  # 归一化的行坐标
        cn = c / (env.cols - 1)  # 归一化的列坐标
        
        for a in range(env.n_actions):
            an = a / (env.n_actions - 1)
            features[s, a] = np.array([
                1.0, 
                rn, 
                cn, 
                an, 
                rn * cn, 
                rn * an, 
                cn * an, 
                rn ** 2, 
                cn ** 2, 
                an ** 2, 
                rn * cn * an,
                rn**2*an,
                cn**2*an,
                rn*an**2,
                cn*rn**2,
                rn**3,
                cn**3,
                an**3,
                ])
    
    return features

def q_value(theta, features, state_idx, action_idx):
    # print(features[state_idx, action_idx])
    # print(theta)
    return features[state_idx, action_idx] @ theta

def sarsa_linear(env: GridWorld, config: SarsaLinearConfig) -> SarsaLinearResult:
    rng = np.random.default_rng(config.seed)
    policy=env.policy
    features = build_features(env)
    theta = np.zeros(features.shape[2])  # 初始化参数
    converged = False
    iterations = 0
    
    for count in range(config.n_episodes):
        state_idx = rng.choice(env.n_states)# s_t
        state = env.index_to_state(state_idx)  
        
        action_idx=rng.choice(env.n_actions, p=policy[state_idx])
        action = env.actions[action_idx]  
        
        for i in range(config.episode_length):
            state_next, reward = env.step(state, action)  # s_{t+1}, r_{t+1}
            state_next_idx = env.state_to_index(state_next)
            
            # 根据 s_{t+1} 的当前策略采样 a_{t+1}
            action_next_idx = rng.choice(env.n_actions, p=policy[state_next_idx])
            action_next = env.actions[action_next_idx]
            
            # TD error
            q_s = q_value(theta, features, state_idx, action_idx)
            q_next = q_value(theta, features, state_next_idx, action_next_idx)
            delta = reward + env.gamma * q_next - q_s
            
            theta += config.alpha * delta * features[state_idx, action_idx]
            
            best_action_idx = np.argmax([q_value(theta, features, state_idx, a_idx) for a_idx in range(env.n_actions)])
            policy[state_idx]=np.full((env.n_actions), config.epsilon/env.n_actions)
            policy[state_idx][best_action_idx] = 1-config.epsilon+config.epsilon/env.n_actions
            
            action = action_next
            state = state_next
            state_idx = state_next_idx
            action_idx = action_next_idx
        
        iterations += 1
    
    return SarsaLinearResult(theta=theta, policy=policy, iterations=iterations)

if __name__ == "__main__":
    env = GridWorld()
    cfg = SarsaLinearConfig()
    
    start_time = time.perf_counter()
    result = sarsa_linear(env, cfg)
    elapsed_time = time.perf_counter() - start_time
    
    # 提取策略并可视化
    policy = result.policy
    env_with_policy = GridWorld(policy=policy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value = env_with_policy.get_true_value_by_policy(policy)
    env_with_policy.render_with_state_value(state_value, title="SARSA Linear V*(s)", ax=ax1)
    env_with_policy.render_with_policy(ax=ax2)
    plt.show()
    
    #将soft policy 转为hard policy，重新查看结果
    best_action_idx=np.argmax(result.policy,axis=1)
    policy=np.array([np.eye(env.n_actions)[best_action_idx[s_idx]] for s_idx in range(env.n_states)])
    env=GridWorld(policy=policy)
    state_value=env.get_true_value_by_policy(policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(state_value, title="Sarsa Value ", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    print("\n=== 运行配置与耗时 ===")
    print("算法: SARSA with Linear Function Approximation")
    print(f"配置参数：{asdict(cfg)}")
    print(f"迭代信息：iterations={result.iterations}")
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
