
"""
增量学习估计状态值，TD（0）
TD(0)是为了估计状态值，即$E[v_\pi(S)]=E[R+\gamma v_\pi(S')]$
设$f(w)=w-E[R+\gamma v_\pi(S')]$
即求解：$f(w)=0$
采用RM算法，设定$\eta=E[R+\gamma v_\pi(S')]-v_\pi(s')=E[R+\gamma v_\pi(S')]-（r+\gamma v_\pi(s'))$
于是$\tilde{f(w,\eta)}=f(w)+\eta=w_k-（r+\gamma v_\pi(s'))$
$w$就是对$v_\pi(S)$的估计，将$w_{k}$作为$v_{k}(S)$
于是计算方法为：$v_{k+1}(s)=v_k(s)-\alpha_k(v_k(s)-(r+\gamma v_\pi(s')))$
在实际应用中由于不知道模型，将$v_\pi(s')$用$v_k(s')$代替,学习率采用一个小值$\alpha_k =\alpha$

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

@dataclass
class TDConfig:
    alpha: float = 0.0005#学习率
    #epsilon: float = 0.1
    max_iteration=3000
    max_episode_length: int = 100
    seed: int = 42

@dataclass
class TDResult:
    v: np.ndarray
    true_v: np.ndarray
    erros: List[float]

def td_state_value(env: GridWorld, config: TDConfig) -> TDResult:
    true_v=env.true_value()
    rng=np.random.RandomState(config.seed)
    v=np.zeros(env.n_states)
    erros=[]

    for count in range(config.max_iteration):
        state_idx=rng.choice(env.n_states)#随机选择一个状态
        for t in range(config.max_episode_length):
            state_next,reward=env.sample_next(env.index_to_state(state_idx),env.policy,rng)#采用环境默认的随机策略
            v[state_idx]=v[state_idx]+config.alpha*(reward+env.gamma*v[env.state_to_index(state_next)]-v[state_idx])
            state_idx=env.state_to_index(state_next)

        # 计算与真实值的均方根误差(RMSE)
        rmse = np.sqrt(np.mean((v - true_v) ** 2))
        erros.append(rmse)

    return TDResult(v,true_v,erros)

def plot_results(env: GridWorld, true_v: np.ndarray, approx_v: np.ndarray, errors: List[float]):
    """绘制ground truth state value、估计值和误差曲线"""
    # 将向量形式的值转换为矩阵形式
    true_v_mat = env.value_vector_to_matrix(true_v)
    approx_v_mat = env.value_vector_to_matrix(approx_v)

    # 创建3个子图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 绘制ground truth state value
    env.render_with_state_value(true_v, title="Ground Truth State Value", use_heatmap=False, ax=axes[0])

    # 绘制估计的state value
    env.render_with_state_value(approx_v, title="Estimated State Value", use_heatmap=False, ax=axes[1])

    # 绘制误差曲线
    axes[2].plot(errors, color="tab:blue")
    axes[2].set_title("TD Convergence Curve")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env=GridWorld()
    config=TDConfig()
    result=td_state_value(env,config)

    # 打印结果
    print("Ground Truth State Value:")
    print(env.value_vector_to_matrix(result.true_v))
    print("\nEstimated State Value:")
    print(env.value_vector_to_matrix(result.v))
    print(f"\nFinal RMSE: {result.erros[-1]:.6f}")

    # 绘制结果
    plot_results(env, result.true_v, result.v, result.erros)
