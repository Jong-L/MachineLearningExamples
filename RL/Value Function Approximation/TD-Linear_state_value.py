'''
策略为环境本身的随机策略
用参数$\theta$的线性函数近似状态值
特征向量为$\phi(s)=\phi(x,y)=[1, x, y, x \cdot y, x^2, y^2]^T$，其中$x$和$y$分别表示状态$s$的行和列坐标。
$\hat{v_\pi(s,\theta)}=\phi(s)^T\cdot \theta$
需要求解最优化问题：
$$
\min J(\theta)=\frac 12 E[(\hat{v_\pi(S,\theta)}-v_\pi(S))^2]
$$
这是一个凸优化问题，只需要求解梯度为0：
$$
\theta^* \quad s.t.\quad  E[\hat{v_\pi(S,\theta)}-v_\pi(S)]\nabla \hat{v_\pi(S,\theta)}=0
$$
进一步，即求解：
$$
g(\theta)=E[\hat{v_\pi(S,\theta)}-v_\pi(S)]\phi(S)=0
$$
采用随机梯度下降求解：
$$
\theta_{t+1}=\theta_t-\alpha_t \phi(s)(\hat{v_\pi(s,\theta)}-v_\pi(s))
$$
在这一节中，$v_\pi(s)$用$r_{t+1}+\gamma \hat{v_\pi}(s_{t+1},\theta)$代替。
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from dataclasses import asdict, dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

@dataclass
class TDLinearStateValueConfig:
    alpha: float = 0.0005
    max_iteration: int = 30000
    seed: int = 0

@dataclass
class TDLinearStateValueResult:
    theta: np.ndarray
    errors: np.ndarray
    approx_v : np.ndarray


def build_features(env):# 构建特征向量phi(s)=phi(x,y)
    features = np.zeros((env.n_states, 6))
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
            cn ** 2
        ])
    
    return features


def td_linear_policy_evaluation(
    env,
    features,
    true_v,
    cfg: TDLinearStateValueConfig,
):
    rng = np.random.default_rng(cfg.seed)
    theta = np.zeros(features.shape[1]) #  初始参数,和特征向量的维度相同
    errors = [] #  用于存储每一步的误差值

    state_idx = rng.choice(env.n_states) #  随机选择一个起始状态
    
    for _ in range(cfg.max_iteration):
        state = env.index_to_state(state_idx)
        state_next, action,reward = env.sample_next(state,None, rng)
        state_next_idx = env.state_to_index(state_next)

        v_s = features[state_idx] @ theta
        v_next = features[state_next_idx] @ theta
        delta = reward + env.gamma * v_next - v_s
        theta += cfg.alpha * delta * features[state_idx]

        approx_v = features @ theta
        rmse = np.sqrt(np.mean((approx_v - true_v) ** 2))
        errors.append(rmse)

        state_idx = state_next_idx

    return TDLinearStateValueResult(
        theta=theta,
        errors=np.array(errors),
        approx_v=approx_v,
    )

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
    env = GridWorld()
    true_v_vec = env.get_true_value_by_policy()
    true_v = true_v_vec.reshape(env.rows, env.cols)
    features = build_features(env)

    cfg = TDLinearStateValueConfig(
        alpha=0.0005,
        max_iteration=40000,
        seed=42,
    )
    start_time = time.perf_counter()
    result = td_linear_policy_evaluation(
        env=env,
        features=features,
        true_v=true_v_vec,
        cfg=cfg,
    )
    elapsed_time = time.perf_counter() - start_time
    theta = result.theta
    errors = result.errors
    approx_v = result.approx_v.reshape(env.rows, env.cols)

    np.set_printoptions(precision=3, suppress=True)
    print("Theoretical state value V_pi:")
    print(true_v)
    print("\nLearned parameter theta:")
    print(theta)
    print(f"\nFinal RMSE: {errors[-1]:.6f}")

    print("\n=== 运行配置与耗时 ===")
    print("算法: TD(0) Linear Function Approximation")
    print(f"配置参数: {asdict(cfg)}")
    print(f"迭代信息: iterations={len(errors)}, final_rmse={errors[-1]:.6f}")
    print(f"算法运行时间: {elapsed_time:.6f} 秒")

    plot_results(true_v, approx_v, errors)

if __name__ == "__main__":
    main()
