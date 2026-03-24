"""
测试MC Exploring Starts算法，并与最优解比较
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld
from MCExploringStarts import MonteCarloExploringStarts, MCESConfig

def test_mces_against_optimal():
    """将MCES结果与理论最优值比较"""
    print("=" * 60)
    print("Testing MC Exploring Starts Against Optimal Solution")
    print("=" * 60)
    
    # 创建环境
    env = GridWorld()
    
    # 计算最优值函数（通过求解线性系统）
    print("\n1. Computing optimal value function via policy iteration...")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from BellamOptimalityEquation.policy_iteration import policy_iteration, PolicyIterationConfig
    result = policy_iteration(env, PolicyIterationConfig())
    optimal_v = result.value
    optimal_policy = result.policy
    print(f"Optimal V(s) range: [{np.min(optimal_v):.4f}, {np.max(optimal_v):.4f}]")
    print(f"Optimal policy shape: {optimal_policy.shape}")
    
    # 运行MC Exploring Starts
    print("\n2. Running Monte Carlo Exploring Starts...")
    config = MCESConfig(
        gamma=0.9,
        max_episodes=20000,
        max_episode_length=200,
        convergence_threshold=1e-6,
        verbose=True,
        plot_interval=5000
    )
    agent = MonteCarloExploringStarts(env, config)
    result = agent.train()
    
    # 比较结果
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)
    
    # 计算值函数的均方误差
    mse = np.mean((result.value - optimal_v) ** 2)
    max_abs_error = np.max(np.abs(result.value - optimal_v))
    
    print(f"MSE between MCES V(s) and optimal V(s): {mse:.6f}")
    print(f"Max absolute error: {max_abs_error:.6f}")
    
    # 检查策略是否一致
    # MCES的策略是确定性的，policy shape: (n_states, n_actions)，每个状态只有一个1
    mces_deterministic = np.argmax(result.policy, axis=1)
    optimal_deterministic = np.argmax(optimal_policy, axis=1)
    
    policy_agreement = np.mean(mces_deterministic == optimal_deterministic)
    print(f"Policy agreement with optimal: {policy_agreement * 100:.1f}%")
    
    # 显示差异最大的状态
    if max_abs_error > 0.01:
        print("\nStates with largest value differences:")
        diffs = np.abs(result.value - optimal_v)
        top_k = min(5, len(diffs))
        indices = np.argsort(diffs)[-top_k:][::-1]
        for idx in indices:
            r, c = divmod(idx, env.cols)
            print(f"  State ({r},{c}): MCES={result.value[idx]:.4f}, Optimal={optimal_v[idx]:.4f}, diff={diffs[idx]:.4f}")
    
    return {
        'mces_v': result.value,
        'optimal_v': optimal_v,
        'mces_policy': mces_deterministic,
        'optimal_policy': optimal_deterministic,
        'mse': mse,
        'max_error': max_abs_error,
        'policy_agreement': policy_agreement
    }

if __name__ == "__main__":
    results = test_mces_against_optimal()