"""
SARSA (on-policy) + 线性函数近似动作值

目标：使用线性近似的动作价值函数 Q_hat(s, a) = theta_a^T * phi(s)
来实现 SARSA。

与伪代码的对应关系：
- 计算 TD 误差：delta = r + gamma * Q_hat(s_{t+1}, a_{t+1}) - Q_hat(s_t, a_t)
- 更新参数（线性近似的梯度为特征向量）：theta_{a_t} <- theta_{a_t} + alpha * delta * phi(s_t)
- 行为选择：使用 epsilon-greedy，使得下一步动作 a_{t+1} 来自当前策略
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld


def build_state_features(env: GridWorld) -> np.ndarray:
    """
    phi(s) = [1, x, y, x*y, x^2, y^2]^T
    其中 (x, y) 为状态坐标的归一化形式（行/列）。
    """
    # env.index_to_state 返回 (r, c)
    features = np.zeros((env.n_states, 6), dtype=float)
    denom_r = max(1, env.rows - 1)
    denom_c = max(1, env.cols - 1)

    for s_idx in range(env.n_states):
        r, c = env.index_to_state(s_idx)
        x = r / denom_r
        y = c / denom_c
        features[s_idx] = np.array(
            [1.0, x, y, x * y, x**2, y**2],
            dtype=float,
        )
    return features


def is_terminal(env: GridWorld, s_idx: int) -> bool:
    return env.index_to_state(s_idx) == env.target


def greedy_action(
    theta: np.ndarray,
    features: np.ndarray,
    s_idx: int,
    rng: np.random.Generator,
) -> int:
    """
    返回 argmax_a Q_hat(s, a)；若并列最大，随机打破平局。
    """
    # theta: (n_actions, n_features), features[s_idx]: (n_features,)
    q_vals = theta @ features[s_idx]
    max_q = np.max(q_vals)
    best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
    return int(rng.choice(best_actions))


def epsilon_greedy_action(
    theta: np.ndarray,
    features: np.ndarray,
    s_idx: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    epsilon-greedy：
    - 以 1-epsilon 的概率选贪心动作
    - 以 epsilon 的概率在非贪心动作中均匀采样
    """
    n_actions = theta.shape[0]
    a_star = greedy_action(theta, features, s_idx, rng)
    if n_actions <= 1:
        return a_star

    if rng.random() < (1.0 - epsilon):
        return a_star

    non_greedy = [a for a in range(n_actions) if a != a_star]
    # 若并列最大导致非贪心定义略有偏差：这里仍然可用 a_star 的某个选取作为“贪心”
    return int(rng.choice(non_greedy))


def q_hat(theta: np.ndarray, features: np.ndarray, s_idx: int, a_id: int) -> float:
    """Q_hat(s,a)=theta[a]^T phi(s)"""
    return float(theta[a_id] @ features[s_idx])


@dataclass
class SARSAConfig:
    num_episodes: int = 3000
    max_steps_per_episode: int = 200
    alpha: float = 0.02
    gamma: float = 0.9
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42


def sarsa_linear_function_approximation(
    env: GridWorld,
    features: np.ndarray,
    config: SARSAConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.seed)

    n_actions = len(env.actions)
    n_features = features.shape[1]
    # 分块参数：每个动作一组权重 theta[a]，对应 phi(s) 的线性组合
    theta = np.zeros((n_actions, n_features), dtype=float)

    # 统计信息
    episode_returns: List[float] = []
    td_abs_mean: List[float] = []

    # 起始状态：避免直接从目标点开始（这样会让 episode 过短）
    all_states = np.arange(env.n_states)
    start_states = np.array([s for s in all_states if not is_terminal(env, int(s))], dtype=int)
    if len(start_states) == 0:
        raise ValueError("环境中没有可用起始状态（target 可能被设为所有状态）。")

    epsilon = config.epsilon_start

    for ep in range(config.num_episodes):
        s_idx = int(rng.choice(start_states))
        a_id = epsilon_greedy_action(theta, features, s_idx, epsilon, rng)

        total_reward = 0.0
        td_errors: List[float] = []

        for _ in range(config.max_steps_per_episode):
            # 取样环境：使用我们选定的动作（不要依赖 env.policy）
            state = env.index_to_state(s_idx)
            action = env.actions[a_id]
            next_state, reward = env.step(state, action)
            s_next_idx = env.state_to_index(next_state)

            total_reward += reward

            q_current = q_hat(theta, features, s_idx, a_id)

            if is_terminal(env, s_next_idx):
                q_next = 0.0
                delta = reward - q_current
                # 线性近似梯度：对参数 theta[a_id] 的梯度就是 phi(s_idx)
                theta[a_id] += config.alpha * delta * features[s_idx]
                td_errors.append(abs(delta))
                break

            a_next = epsilon_greedy_action(theta, features, s_next_idx, epsilon, rng)
            q_next = q_hat(theta, features, s_next_idx, a_next)

            delta = reward + config.gamma * q_next - q_current
            theta[a_id] += config.alpha * delta * features[s_idx]
            td_errors.append(abs(delta))

            s_idx = s_next_idx
            a_id = a_next

        episode_returns.append(total_reward)
        td_abs_mean.append(float(np.mean(td_errors) if td_errors else 0.0))

        # epsilon 衰减：近似“策略改进后逐步变得更贪心”
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    history = {
        "episode_returns": np.asarray(episode_returns, dtype=float),
        "td_abs_mean": np.asarray(td_abs_mean, dtype=float),
    }
    return theta, history


def greedy_state_value(theta: np.ndarray, features: np.ndarray, env: GridWorld) -> np.ndarray:
    """
    从 Q_hat 计算贪心状态价值：V_hat(s)=max_a Q_hat(s,a)
    """
    q_vals = (theta @ features.T).T  # shape: (n_states, n_actions)
    v_hat = np.max(q_vals, axis=1)
    return v_hat.reshape(env.rows, env.cols)


def build_epsilon_greedy_policy_from_theta(
    env: GridWorld,
    theta: np.ndarray,
    features: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    构建 env.policy：用于 render_with_policy 可视化。
    """
    policy = np.zeros((env.n_states, len(env.actions)), dtype=float)
    rng = np.random.default_rng(0)

    for s_idx in range(env.n_states):
        if is_terminal(env, s_idx):
            # 目标点不参与策略渲染：给均匀分布避免箭头“消失”
            policy[s_idx] = 1.0 / len(env.actions)
            continue

        # 采用“某个具体贪心动作”为 a_star（并列情况下随机打破）
        a_star = greedy_action(theta, features, s_idx, rng)
        n_actions = len(env.actions)
        if n_actions <= 1:
            policy[s_idx, a_star] = 1.0
        else:
            for a_id in range(n_actions):
                if a_id == a_star:
                    policy[s_idx, a_id] = 1.0 - epsilon
                else:
                    policy[s_idx, a_id] = epsilon / (n_actions - 1)
    return policy


def main() -> None:
    env = GridWorld(gamma=0.9)
    features = build_state_features(env)

    config = SARSAConfig(
        num_episodes=4000,
        max_steps_per_episode=200,
        alpha=0.02,
        gamma=env.gamma,
        epsilon_start=0.2,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=42,
    )

    theta, history = sarsa_linear_function_approximation(env, features, config)

    v_hat = greedy_state_value(theta, features, env)

    print("Training finished.")
    print(f"Final mean |TD error|: {history['td_abs_mean'][-1]:.6f}")
    print(f"Final episode return: {history['episode_returns'][-1]:.3f}")
    print("Learned theta (shape = n_actions x n_features):", theta.shape)

    # 可视化
    plt.figure(figsize=(7, 4))
    plt.plot(history["td_abs_mean"], label="mean(|TD error|)")
    plt.xlabel("Episode")
    plt.ylabel("mean(|delta|)")
    plt.title("SARSA (linear Q) convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    env.render_with_state_value(
        v_hat,
        title="V_hat(s) = max_a Q_hat(s,a) (SARSA linear Q)",
        use_heatmap=True,
    )

    env.policy = build_epsilon_greedy_policy_from_theta(
        env=env,
        theta=theta,
        features=features,
        epsilon=config.epsilon_end,
    )
    env.render_with_policy()


if __name__ == "__main__":
    main()



