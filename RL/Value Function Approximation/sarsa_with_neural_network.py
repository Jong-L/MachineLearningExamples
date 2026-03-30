"""
SARSA with Neural Network Function Approximation
在 SARSA 算法中使用神经网络近似 Q 值
使用简单的全连接网络来逼近 Q(s, a)

AI编写，用于比较和线性函数近似 Q 值的效果，发现也不太好，暂且保留。
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
class SarsaNNConfig:
    alpha: float = 0.001  # 学习率
    epsilon: float = 0.1  # epsilon greedy policy
    n_episodes: int = 10000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100  # 连续 patience 次参数变化小于 threshold 则认为收敛
    hidden_size: int = 32  # 隐藏层大小
    gamma: float = 0.99  # 折扣因子

@dataclass
class SarsaNNResult:
    model_weights: list  # 神经网络权重
    policy: np.ndarray
    iterations: int
    converged: bool


class SimpleQNetwork:
    """简单的全连接网络用于近似 Q 值
    
    输入：状态特征 + 动作特征 (拼接向量)
    输出：单个 Q 值
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        self.rng = rng
        
        # Xavier 初始化
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # 保存中间结果用于反向传播
        self.cache = {}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """前向传播
        
        Parameters:
            x: 输入特征向量 (batch_size, input_size)
        
        Returns:
            q_value: Q 值预测 (batch_size, output_size)
        """
        # 第一层
        self.cache['z1'] = x @ self.W1 + self.b1
        self.cache['a1'] = self.relu(self.cache['z1'])
        
        # 第二层
        self.cache['z2'] = self.cache['a1'] @ self.W2 + self.b2
        q_value = self.cache['z2']
        
        return q_value
    
    def backward(self, x, y_target, learning_rate):
        """反向传播，更新权重
        
        Parameters:
            x: 输入特征 (batch_size, input_size)
            y_target: 目标 Q 值 (batch_size, output_size)
            learning_rate: 学习率
        
        Returns:
            loss: 均方误差损失
        """
        batch_size = x.shape[0]
        
        # 前向传播
        y_pred = self.forward(x)
        
        # 计算损失 (MSE)
        loss = np.mean((y_pred - y_target) ** 2)
        
        # 反向传播
        # 输出层梯度
        dz2 = (y_pred - y_target) * (2.0 / batch_size)  # (batch_size, output_size)
        dW2 = self.cache['a1'].T @ dz2  # (hidden_size, output_size)
        db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, output_size)
        
        # 隐藏层梯度
        da1 = dz2 @ self.W2.T  # (batch_size, hidden_size)
        dz1 = da1 * self.relu_derivative(self.cache['z1'])  # (batch_size, hidden_size)
        dW1 = x.T @ dz1  # (input_size, hidden_size)
        db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, hidden_size)
        
        # 梯度裁剪（防止梯度爆炸）
        max_grad_norm = 1.0
        for grad in [dW1, db1, dW2, db2]:
            norm = np.linalg.norm(grad)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm
        
        # 参数更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
        return loss
    
    def get_weights(self):
        """获取网络权重"""
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]
    
    def set_weights(self, weights):
        """设置网络权重"""
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in weights]


def build_state_features(env):
    """构建状态特征向量
    
    每个状态用一个 4 维特征向量表示：
    - 归一化的行坐标
    - 归一化的列坐标
    - 行坐标的平方
    - 列坐标的平方
    """
    features = np.zeros((env.n_states, 4))
    
    for s in range(env.n_states):
        r, c = env.index_to_state(s)
        rn = r / (env.rows - 1)  # 归一化的行坐标
        cn = c / (env.cols - 1)  # 归一化的列坐标
        
        features[s] = np.array([rn, cn, rn ** 2, cn ** 2])
    
    return features


def build_state_action_features(env, state_features):
    """构建状态 - 动作特征向量
    
    将状态特征和动作 one-hot 编码拼接
    """
    features = np.zeros((env.n_states, env.n_actions, state_features.shape[1] + env.n_actions))
    
    for s in range(env.n_states):
        for a in range(env.n_actions):
            # 动作的 one-hot 编码
            action_onehot = np.zeros(env.n_actions)
            action_onehot[a] = 1.0
            
            # 拼接状态特征和动作特征
            features[s, a] = np.concatenate([state_features[s], action_onehot])
    
    return features


def sarsa_nn(env: GridWorld, config: SarsaNNConfig) -> SarsaNNResult:
    rng = np.random.default_rng(config.seed)
    policy = env.policy.copy()
    
    # 构建特征
    state_features = build_state_features(env)
    state_action_features = build_state_action_features(env, state_features)
    input_size = state_action_features.shape[2]
    
    # 初始化神经网络
    q_network = SimpleQNetwork(input_size, config.hidden_size, rng=rng)
    
    # 保存初始权重用于收敛判断
    initial_weights = q_network.get_weights()
    last_weights = [w.copy() for w in initial_weights]
    
    no_improvement_count = 0
    converged = False
    iterations = 0
    
    total_loss = 0
    loss_count = 0
    
    for count in range(config.n_episodes):
        # 随机选择初始状态
        state_idx = rng.choice(env.n_states)
        state = env.index_to_state(state_idx)
        
        # 根据当前策略选择动作
        action_idx = rng.choice(env.n_actions, p=policy[state_idx])
        action = env.actions[action_idx]
        
        episode_loss = 0
        
        for i in range(config.episode_length):
            # 执行动作，观察下一状态和奖励
            state_next, reward = env.step(state, action)
            state_next_idx = env.state_to_index(state_next)
            
            # 根据 s_{t+1} 的当前策略采样 a_{t+1} (on-policy)
            action_next_idx = rng.choice(env.n_actions, p=policy[state_next_idx])
            action_next = env.actions[action_next_idx]
            
            # 计算当前 Q 值和目标 Q 值
            current_features = state_action_features[state_idx, action_idx].reshape(1, -1)
            q_current = q_network.forward(current_features)[0, 0]
            
            next_features = state_action_features[state_next_idx, action_next_idx].reshape(1, -1)
            q_next = q_network.forward(next_features)[0, 0]
            
            # TD 目标 (SARSA)
            td_target = reward + config.gamma * q_next
            
            # 训练网络 (单样本 SGD)
            loss = q_network.backward(current_features, np.array([[td_target]]), config.alpha)
            episode_loss += loss
            
            # 基于新的 Q 值估计更新策略 (epsilon-greedy)
            q_values = []
            for a_idx in range(env.n_actions):
                feat = state_action_features[state_idx, a_idx].reshape(1, -1)
                q_values.append(q_network.forward(feat)[0, 0])
            
            best_action_idx = np.argmax(q_values)
            policy[state_idx] = np.full(env.n_actions, config.epsilon / env.n_actions)
            policy[state_idx][best_action_idx] = 1 - config.epsilon + config.epsilon / env.n_actions
            
            # 转移到下一步
            state = state_next
            state_idx = state_next_idx
            action = action_next
            action_idx = action_next_idx
        
        total_loss += episode_loss
        loss_count += 1
        
        # 检查收敛性
        iterations += 1
        
        # 每 100 个 episode 检查一次权重变化
        if count % 100 == 0 and count > 0:
            current_weights = q_network.get_weights()
            max_weight_change = max(
                np.max(np.abs(current_weights[i] - last_weights[i]))
                for i in range(len(current_weights))
            )
            
            if max_weight_change < config.threshold:
                no_improvement_count += 1
                if no_improvement_count >= config.patience:
                    converged = True
            else:
                no_improvement_count = 0
            
            last_weights = [w.copy() for w in current_weights]
    
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    print(f"\n平均训练损失：{avg_loss:.6f}")
    
    return SarsaNNResult(
        model_weights=q_network.get_weights(),
        policy=policy,
        iterations=iterations,
        converged=converged
    )


if __name__ == "__main__":
    env = GridWorld()
    cfg = SarsaNNConfig(
        alpha=0.001,
        epsilon=0.1,
        n_episodes=10000,
        episode_length=100,
        seed=42,
        hidden_size=32,
        gamma=0.99
    )
    
    start_time = time.perf_counter()
    result = sarsa_nn(env, cfg)
    elapsed_time = time.perf_counter() - start_time
    
    # 提取策略并可视化
    policy = result.policy
    env_with_policy = GridWorld(policy=policy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value = env_with_policy.get_true_value_by_policy(policy)
    env_with_policy.render_with_state_value(state_value, title="SARSA-NN V*(s)", ax=ax1)
    env_with_policy.render_with_policy(ax=ax2)
    plt.show()
    
    # 将 soft policy 转为 hard policy，重新查看结果
    best_action_idx = np.argmax(result.policy, axis=1)
    policy = np.array([np.eye(env.n_actions)[best_action_idx[s_idx]] for s_idx in range(env.n_states)])
    env = GridWorld(policy=policy)
    state_value = env.get_true_value_by_policy(policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(state_value, title="SARSA-NN Value", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    print("\n=== 运行配置与耗时 ===")
    print("算法: SARSA with Neural Network Function Approximation")
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
        
        print(f"\n总状态值：{result_value_sum:.5f}, 最优总状态值：{true_value_sum:.5f}")
        print(f"mean abs error: {error:.5f}")
