'''
用参数$\theta$的线性函数近似状态值
特征向量为$\phi(s)=\phi(x,y)=[1, x, y, x \cdot y, x^2, y^2]^T$，其中$x$和$y$分别表示状态$s$的行和列坐标。
1. 初始化参数向量 $\theta$ 为零向量
2. 对于每个时间步：
   - 选择当前状态 $s$
   - 执行动作，观察下一状态 $s'$ 和奖励 $r$
   - 计算TD误差：
     $$\delta = r + \gamma \cdot V(s') - V(s)$$
     其中 $V(s) = \phi(s)^T \theta$
   - 更新参数：
     $$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \phi(s)$$
   - 计算并记录近似价值函数与真实价值函数之间的均方根误差(RMSE)

其中：
- $\alpha$：学习率
- $\gamma$：折扣因子
- $\phi(s)$：状态s的特征向量
- $V(s)$：状态s的近似价值函数

'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

def build_features(env):
    features = np.zeros((env.n_states, 6))# 特征向量
    for s in range(env.n_states):
        r, c = env.index_to_state(s)
        rn = r / (env.rows - 1) #  计算归一化的行和列坐标
        cn = c / (env.cols - 1)

        features[s] = np.array([
            1.0,
            rn,
            cn,
            rn * cn,
            rn ** 2,
            cn ** 2,
        ])
    return features


def td_linear_policy_evaluation(
    env,
    features,
    true_v,
    alpha=0.0005,
    num_steps=30000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    theta = np.zeros(features.shape[1]) #  初始参数
    errors = [] #  用于存储每一步的误差值

    valid_starts = [
        env.state_to_index((r, c))
        for r in range(env.rows)
        for c in range(env.cols)
    ]

    state_idx = rng.choice(valid_starts) #  随机选择一个起始状态
    
    for _ in range(num_steps):
        state = env.index_to_state(state_idx)
        next_state, reward = env.sample_next(state, rng)
        next_idx = env.state_to_index(next_state)

        v_s = features[state_idx] @ theta
        v_next = features[next_idx] @ theta
        delta = reward + env.gamma * v_next - v_s
        theta += alpha * delta * features[state_idx]

        approx_v = features @ theta
        rmse = np.sqrt(np.mean((approx_v - true_v) ** 2))
        errors.append(rmse)

        state_idx = next_idx

    return theta, np.array(errors)

def plot_results(true_v, approx_v, errors):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].imshow(true_v, cmap="coolwarm")
    axes[0].set_title("Theoretical State Value")
    axes[0].set_xticks(range(true_v.shape[1]))
    axes[0].set_yticks(range(true_v.shape[0]))
    for i in range(true_v.shape[0]):
        for j in range(true_v.shape[1]):
            axes[0].text(j, i, f"{true_v[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(approx_v, cmap="coolwarm")
    axes[1].set_title("TD Linear Approximation")
    axes[1].set_xticks(range(approx_v.shape[1]))
    axes[1].set_yticks(range(approx_v.shape[0]))
    for i in range(approx_v.shape[0]):
        for j in range(approx_v.shape[1]):
            axes[1].text(j, i, f"{approx_v[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].plot(errors, color="tab:blue")
    axes[2].set_title("TD Convergence Curve")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    env = GridWorld(gamma=0.9)
    true_v_vec = env.get_true_value_by_policy()
    true_v = true_v_vec.reshape(env.rows, env.cols)
    features = build_features(env)

    theta, errors = td_linear_policy_evaluation(
        env=env,
        features=features,
        true_v=true_v_vec,
        alpha=0.0005,
        num_steps=50000,
        seed=42,
    )

    approx_v = (features @ theta).reshape(env.rows, env.cols)

    np.set_printoptions(precision=3, suppress=True)
    print("Theoretical state value V_pi:")
    print(true_v)
    print("\nLearned parameter theta:")
    print(theta)
    print(f"\nFinal RMSE: {errors[-1]:.6f}")

    plot_results(true_v, approx_v, errors)

if __name__ == "__main__":
    main()
