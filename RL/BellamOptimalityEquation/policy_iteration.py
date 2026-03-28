"""
由于整个示例中采用的环境比较简单，直接根据策略求解状态值对算力要求也不高，所以也给出策略迭代的示例
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld
from optimal_solution_manager import (
    generate_env_id,
    save_optimal_solution,
    OptimalSolution
)

@dataclass
class PolicyIterationConfig:
    threshold: float = 1e-6
    max_iter: int = 1000

@dataclass
class PolicyIterationResult:
    value: np.ndarray
    policy: np.ndarray
    iterations: int
    delta: float
    converged: bool
    env_id: Optional[str] = None  # 环境ID，用于标识解对应的环境

def q_value(env: GridWorld, v: np.ndarray, state_idx: int, action: Tuple[int, int]):
    state = env.index_to_state(state_idx)
    next_state, reward = env.step(state, action)
    next_state_idx = env.state_to_index(next_state)
    return reward + env.gamma * v[next_state_idx]

def policy_iteration(env: GridWorld, cfg: PolicyIterationConfig):
    policy=np.ones((env.n_states, env.n_actions))/env.n_actions#初始化策略为均匀分布
    v=np.zeros(env.n_states, dtype=float)
    delta = np.inf
    converged = False
    iterations = 0

    for count in range(cfg.max_iter):
        #policy evaluation
        new_v=env.get_true_value_by_policy(policy)#与值迭代的区别在于这里是求解状态值

        #policy improvement
        for s_idx in range(env.n_states):
            q_vals = np.array([q_value(env, new_v, s_idx, action) for action in env.actions], dtype=float)#所有动作的q值
            best_a = int(np.argmax(q_vals))
            policy[s_idx] = np.eye(env.n_actions)[best_a]
        
        delta = np.max(np.abs(v - new_v))
        v = new_v
        iterations = count + 1
        if delta < cfg.threshold:
            converged = True
            break

    return PolicyIterationResult(
        value=v,
        policy=policy,
        iterations=iterations,
        delta=delta,
        converged=converged,
    )

if __name__ == "__main__":
    start_time = time.perf_counter()
    env = GridWorld()
    cfg = PolicyIterationConfig()
    result=policy_iteration(env, cfg)
    
    elapsed_time = time.perf_counter() - start_time
    print("\n=== 运行配置与耗时 ===")
    print("算法: Policy Iteration")
    print(f"配置参数: {asdict(cfg)}")
    print(f"迭代信息: iterations={result.iterations}, converged={result.converged}, delta={result.delta:.20e}")
    print(f"程序运行时间: {elapsed_time:.6f} 秒")

    env = GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(result.value, title="Policy Iteration V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()


    save_result: bool = False
    # 保存最优解
    if save_result:
        # 生成环境ID
        env_id = generate_env_id(
        rows=env.rows,
        cols=env.cols,
        gamma=env.gamma,
        target=env.target,
        forbidden=env.forbidden,
        r_boundary=env.r_boundary,
        r_forbidden=env.r_forbidden,
        r_target=env.r_target
        )

        solution = OptimalSolution(
            env_id=env_id,
            env_config={
                'rows': env.rows,
                'cols': env.cols,
                'gamma': env.gamma,
                'target': env.target,
                'forbidden': list(env.forbidden),
                'r_boundary': env.r_boundary,
                'r_forbidden': env.r_forbidden,
                'r_target': env.r_target
            },
            value=env.get_true_value_by_policy(result.policy),
            policy=result.policy,
            algorithm='policy_iteration',
            timestamp=datetime.now().isoformat(),
            iterations=result.iterations,
            delta=result.delta,
            converged=result.converged
        )
        save_path = save_optimal_solution(solution)
        print(f"最优解已保存到: {save_path}")
