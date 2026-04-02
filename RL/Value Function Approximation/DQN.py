"""
Deep Q-Network with Experience Replay
"""

import os
# 解决 OpenMP 运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple
from collections import deque
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    print("未检测到 PyTorch，将使用 NumPy 实现简单的神经网络")


@dataclass
class DQNConfig:
    """DQN 配置参数"""
    learning_rate: float = 0.001
    gamma: float = 0.9  # 折扣因子
    epsilon: float = 1.0  # 探索率
    epsilon_min: float = 0.01  # 最小探索率
    epsilon_decay: float = 0.995  # 探索率衰减
    replay_buffer_size: int = 10000  # 经验回放缓冲区大小
    batch_size: int = 32  # 批次大小
    target_update_freq: int = 10  # target network 更新频率
    state_dim: int = 25  # 状态维度 (grid world 状态数)
    action_dim: int = 5  # 动作维度
    hidden_dim: int = 64  # 隐藏层维度


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)
        self.rng=np.random.default_rng()
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样批次"""
        batch:List[Tuple] = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class NeuralNetwork:
    """使用 NumPy 实现的简单神经网络（当没有 PyTorch 时使用）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3
    
    def backward(self, X, y_pred, y_true, learning_rate):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差
        delta3 = (y_pred - y_true) / m
        dW3 = self.a2.T @ delta3
        db3 = np.sum(delta3, axis=0, keepdims=True)
        
        # 隐藏层 2 误差
        delta2 = (delta3 @ self.W3.T) * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # 隐藏层 1 误差
        delta1 = (delta2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # 更新权重
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def copy_weights_from(self, other):
        """从另一个网络复制权重"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class TorchDQN(nn.Module):
    """使用 PyTorch 实现的 DQN 网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TorchDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """DQN Agent"""
    def __init__(self, config: DQNConfig):
        self.config = config
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.epsilon = config.epsilon
        
        if USE_TORCH:
            # 使用 PyTorch
            self.main_network = TorchDQN(config.state_dim, config.hidden_dim, config.action_dim)
            self.target_network = TorchDQN(config.state_dim, config.hidden_dim, config.action_dim)
            self.target_network.load_state_dict(self.main_network.state_dict())
            self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)
            self.criterion = nn.MSELoss()
            self.use_torch = True
        else:
            # 使用 NumPy
            self.main_network = NeuralNetwork(config.state_dim, config.hidden_dim, config.action_dim)
            self.target_network = NeuralNetwork(config.state_dim, config.hidden_dim, config.action_dim)
            self.target_network.copy_weights_from(self.main_network)
            self.use_torch = False
    
    def get_q_values(self, state, network="main"):
        """获取 Q 值"""
        # 将状态转换为 one-hot 向量
        if isinstance(state, tuple):
            state_idx = self.config.state_dim
            # 如果 state 是元组，转换为索引
            if hasattr(self, 'env'):
                state_idx = self.env.state_to_index(state)
            else:
                # 对于 grid world，假设状态是 (row, col)
                row, col = state
                # 需要知道环境的 cols，这里使用一个通用方法
                # 实际上我们应该传入环境对象或者预先知道状态维度
                # 简单处理：如果是 tuple，第一个元素*5+第二个元素（针对 5x5 网格）
                state_idx = int(row * 5 + col)
            
            state_vector = np.zeros(self.config.state_dim)
            state_vector[state_idx] = 1.0
        elif isinstance(state, int):
            # 如果 state 是索引，转换为 one-hot
            state_vector = np.zeros(self.config.state_dim)
            state_vector[state] = 1.0
        else:
            # 已经是向量形式
            state_vector = state
        
        if self.use_torch:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            with torch.no_grad():
                if network == "main":
                    q_values = self.main_network(state_tensor)
                else:
                    q_values = self.target_network(state_tensor)
            return q_values.numpy()[0]
        else:
            state_vector = state_vector.reshape(1, -1)
            if network == "main":
                return self.main_network.forward(state_vector)[0]
            else:
                return self.target_network.forward(state_vector)[0]
    
    def select_action(self, state, env):
        """ε-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, env.n_actions - 1)
        else:
            # 利用：选择最优动作
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, env=None):
        """从经验回放中学习"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # 将状态转换为 one-hot 向量
        def to_one_hot(state, dim):
            if isinstance(state, tuple):
                # 假设是 grid world 的 (row, col) 格式
                # 需要根据实际环境调整
                idx = int(state[0] * 5 + state[1])  # 针对 5x5 网格
            else:
                idx = int(state)
            vec = np.zeros(dim)
            vec[idx] = 1.0
            return vec
        
        states_onehot = np.array([to_one_hot(s, self.config.state_dim) for s in states])
        next_states_onehot = np.array([to_one_hot(s, self.config.state_dim) for s in next_states])
        
        if self.use_torch:
            # PyTorch 实现
            states_tensor = torch.FloatTensor(states_onehot)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states_onehot)
            dones_tensor = torch.FloatTensor(dones)
            
            # 计算当前 Q 值
            current_q_values = self.main_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            ).squeeze(1)
            
            # 计算目标 Q 值（使用 target network）
            with torch.no_grad():
                next_q_values = self.target_network(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + self.config.gamma * next_q_values * (1 - dones_tensor)
            
            # 计算损失并优化
            loss = self.criterion(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        else:
            # NumPy 实现
            # 计算当前 Q 值
            current_q_values = self.main_network.forward(states_onehot)
            
            # 计算目标 Q 值
            next_q_values = self.target_network.forward(next_states_onehot)
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            
            # 构建目标值
            targets = current_q_values.copy()
            for i in range(len(actions)):
                target = rewards[i] + self.config.gamma * max_next_q_values[i][0] * (1 - dones[i])
                targets[i, actions[i]] = target
            
            # 反向传播更新
            self.main_network.backward(states_onehot, current_q_values, targets, self.config.learning_rate)
            
            return np.mean((targets - current_q_values) ** 2)
    
    def update_target_network(self):
        """更新 target network"""
        if self.use_torch:
            self.target_network.load_state_dict(self.main_network.state_dict())
        else:
            self.target_network.copy_weights_from(self.main_network)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)


def train_dqn(env, config: DQNConfig, n_episodes: int = 1000, render: bool = False):
    """训练 DQN agent"""
    agent = DQNAgent(config)
    rewards_history = []
    loss_history = []
    
    print(f"开始训练 DQN...")
    print(f"使用框架：{'PyTorch' if USE_TORCH else 'NumPy'}")
    print(f"总 episode 数：{n_episodes}")
    print(f"经验回放缓冲区大小：{config.replay_buffer_size}")
    print(f"批次大小：{config.batch_size}")
    print(f"Target network 更新频率：{config.target_update_freq}")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        # 重置环境
        state = env.index_to_state(0)  # 从左上角开始
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 200  # 限制每个 episode 的最大步数
        
        while not done and step_count < max_steps:
            # 选择动作
            action = agent.select_action(state, env)
            
            # 执行动作
            next_state, reward = env.step(state, env.actions[action])
            
            # 检查是否终止
            done = (next_state == env.target)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 从经验回放中学习
            loss = agent.update()
            if loss is not None:
                loss_history.append(loss)
        
        rewards_history.append(total_reward)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 定期更新 target network
        if episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            elapsed_time = time.time() - start_time
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"平均奖励：{avg_reward:.2f}, "
                  f"ε: {agent.epsilon:.3f}, "
                  f"耗时：{elapsed_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时：{total_time:.1f}s")
    
    return agent, rewards_history, loss_history


def plot_results(rewards_history, loss_history=None):
    """绘制训练结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制奖励曲线
    axes[0].plot(rewards_history)
    axes[0].plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'), 
                 'r--', label='Moving Average (100)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制损失曲线
    if loss_history:
        axes[1].plot(loss_history, alpha=0.5)
        if len(loss_history) > 100:
            axes[1].plot(np.convolve(loss_history, np.ones(100)/100, mode='valid'), 
                         'r--', label='Moving Average (100)')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, env, n_episodes: int = 100):
    """评估训练好的 agent"""
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.index_to_state(0)
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            # 不使用探索（纯利用）
            q_values = agent.get_q_values(state)
            action = np.argmax(q_values)
            
            next_state, reward = env.step(state, env.actions[action])
            state = next_state
            total_reward += reward
            step_count += 1
            done = (next_state == env.target)
        
        total_rewards.append(total_reward)
    
    print(f"\n评估结果 ({n_episodes} episodes):")
    print(f"平均奖励：{np.mean(total_rewards):.2f}")
    print(f"标准差：{np.std(total_rewards):.2f}")
    print(f"最高奖励：{np.max(total_rewards):.2f}")
    print(f"最低奖励：{np.min(total_rewards):.2f}")
    
    return total_rewards


def extract_policy_from_q(agent, env):
    """从 Q 值中提取确定性策略（greedy policy）"""
    policy = np.zeros((env.n_states, env.n_actions))
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            q_values = agent.get_q_values(state)
            best_action = np.argmax(q_values)
            s_idx = env.state_to_index(state)
            policy[s_idx, best_action] = 1.0
    
    return policy


def render_with_value_and_policy(agent, env, title: str = "Learned Value and Policy"):
    """同时绘制状态值和策略图"""
    # 计算每个状态的最大 Q 值（状态值）
    q_values_matrix = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            q_values = agent.get_q_values(state)
            q_values_matrix[r, c] = np.max(q_values)
    
    # 提取最优策略
    optimal_policy = extract_policy_from_q(agent, env)
    
    # 创建带策略的环境
    env_with_policy = GridWorld(policy=optimal_policy)

    state_value=env_with_policy.get_true_value_by_policy()
    
    # 绘制两个子图：左边状态值，右边策略
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：状态值
    env_with_policy.render_with_state_value(
        state_value,  # 转换为一维向量
        title=f"{title} - V*(s)", 
        ax=ax1
    )
    
    # 右图：策略
    env_with_policy.render_with_policy(ax=ax2)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorld(rows=5, cols=5, gamma=0.9)
    config = DQNConfig()
    agent, rewards, losses = train_dqn(env, config, n_episodes=1000)
    
    plot_results(rewards, losses)
    
    evaluate_agent(agent, env, n_episodes=100)
    
    print("\n可视化学习到的状态值和策略...")
    render_with_value_and_policy(agent, env, title="DQN: Learned State Values and Optimal Policy")

