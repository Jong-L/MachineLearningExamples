"""
Deep Q-Network with Experience Replay
使用 neturalwork/BP_multilayer_network_improved.py 中的 BPNetwork 实现
"""

import os
# 解决 OpenMP 运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'neturalwork'))
from BP_multilayer_network_improved import BPNetwork


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
        self.rng = np.random.default_rng()
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样批次"""
        batch: List[Tuple] = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config: DQNConfig, env: GridWorld):
        self.config = config
        self.env = env
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.epsilon = config.epsilon
        
        # 使用 BPNetwork 替代 TorchDQN
        # 网络结构: [state_dim, hidden_dim, hidden_dim, action_dim]
        layer_sizes = [config.state_dim, config.hidden_dim, config.hidden_dim, config.action_dim]
        
        self.main_network = BPNetwork(
            layer_sizes=layer_sizes,
            eta=config.learning_rate,
            activation='relu',
            output_activation='linear'  # DQN 输出 Q 值，回归问题使用线性激活
        )
        
        self.target_network = BPNetwork(
            layer_sizes=layer_sizes,
            eta=config.learning_rate,
            activation='relu',
            output_activation='linear'
        )
        # 同步 target network 参数
        self.target_network.copy_parameters_from(self.main_network)
    
    def get_q_values(self, state, network="main"):
        """向前传播获取 Q 值"""
        # 将状态转换为 one-hot 向量
        state_idx = self.env.state_to_index(state)
        state_vector = np.zeros(self.config.state_dim)
        state_vector[state_idx] = 1.0
        
        # BPNetwork 输入格式为 (input_dim, batch_size)，单个样本为 (input_dim, 1)
        state_input = state_vector.reshape(-1, 1)
        
        if network == "main":
            q_values = self.main_network.predict(state_input)
        else:
            q_values = self.target_network.predict(state_input)
        
        return q_values.flatten()  # 返回一维数组 (action_dim,)
    
    def select_action(self, state) -> int:
        """ε-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.n_actions - 1)
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, env=None):
        """从经验回放中学习"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # 将状态转换为 one-hot 向量
        def to_one_hot(state, dim):
            idx = self.env.state_to_index(state)
            vec = np.zeros(dim)
            vec[idx] = 1.0
            return vec
        
        # 形状: (batch_size, state_dim)
        states_onehot = np.array([to_one_hot(s, self.config.state_dim) for s in states])
        next_states_onehot = np.array([to_one_hot(s, self.config.state_dim) for s in next_states])
        
        # BPNetwork 输入格式为 (input_dim, batch_size)，需要转置
        states_input = states_onehot.T  # (state_dim, batch_size)
        next_states_input = next_states_onehot.T  # (state_dim, batch_size)
        
        # 计算当前网络对每个动作的 Q 值，形状: (action_dim, batch_size)
        current_q = self.main_network.predict(states_input)
        
        # 计算 target network 对 next_state 的 Q 值，形状: (action_dim, batch_size)
        next_q = self.target_network.predict(next_states_input)
        next_q_max = np.max(next_q, axis=0)  # (batch_size,)
        
        # 构造 target：只更新选中动作对应的 Q 值，其余保持当前值不变（梯度为 0）
        targets = current_q.copy()  # (action_dim, batch_size)
        for i in range(self.config.batch_size):
            targets[actions[i], i] = rewards[i] + self.config.gamma * next_q_max[i] * (1 - dones[i])
        
        # 单步更新 main network
        loss = self.main_network.update_batch(states_input, targets)
        
        return loss
    
    def update_target_network(self):
        """更新 target network"""
        self.target_network.copy_parameters_from(self.main_network)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)


def train_dqn(env: GridWorld, config: DQNConfig, n_episodes: int = 1000, render: bool = False):
    """训练 DQN agent"""
    agent = DQNAgent(config, env)
    rewards_history = []
    loss_history = []
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        # 重置环境
        state = env.index_to_state(0)  # 从左上角开始
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 200  # 每个 episode 的最大步数
        
        while not done and step_count < max_steps:
            # 选择动作
            action = agent.select_action(state)
            
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
            loss = agent.update(env)
            if loss is not None:
                loss_history.append(loss)
        
        rewards_history.append(total_reward)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 定期更新 target network
        if episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        # 打印进度
        if (episode + 1) % 500 == 0:
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
            action = int(np.argmax(q_values))
            
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
            best_action = int(np.argmax(q_values))
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
    
    state_value = env_with_policy.get_true_value_by_policy()
    
    # 绘制两个子图：左边状态值，右边策略
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：状态值
    env_with_policy.render_with_state_value(
        state_value,
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
    agent, rewards, losses = train_dqn(env, config, n_episodes=50000)
    
    plot_results(rewards, losses)
    
    evaluate_agent(agent, env, n_episodes=100)
    
    print("\n可视化学习到的状态值和策略...")
    render_with_value_and_policy(agent, env, title="DQN (BPNetwork): Learned State Values and Optimal Policy")
