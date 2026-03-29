"""

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
class NStepSARSAConfig:
    alpha: float = 0.005
    epsilon: float = 0.1
    n_episodes: int = 1000
    episode_length: int = 100
    seed: int = 42
    threshold: float = 1e-6
    patience: int = 100
    n_steps: int = 10#n-step SARSA中的n

@dataclass
class NStepSARSAResult:
    policy: np.ndarray
    q_table: np.ndarray
    converged: bool
    iterations: int

def n_step_sarsa(env: GridWorld, config: NStepSARSAConfig) -> NStepSARSAResult:
    rng=np.random.default_rng(config.seed)
    policy=env.policy
    q_table=np.zeros((env.n_states,env.n_actions))
    last_q_table=np.zeros((env.n_states,env.n_actions))
    no_improvement_count=0
    converged=False
    iterations=0

    for count in range(config.n_episodes):
        state_idx=rng.choice(env.n_states)
        state=env.index_to_state(state_idx)
        action_idx=rng.choice(env.n_actions, p=policy[state_idx])
        action=env.actions[action_idx]
        n_step_episode=[]
        for t in range(config.episode_length):
            state_next,reward=env.step(state,action)#s_{t+1},r_{t+1}
            state_idx=env.state_to_index(state)
            action_idx=env.actions.index(action)
            state_next_idx=env.state_to_index(state_next)

            action_next_idx = rng.choice(env.n_actions, p=policy[state_next_idx])
            action_next = env.actions[action_next_idx]

            n_step_episode.append((state, action, reward))

            if len(n_step_episode)==config.n_steps:
                s_t=env.state_to_index(n_step_episode[0][0])
                a_t=env.actions.index(n_step_episode[0][1])
                n_step_r=np.sum([n_step_episode[i][2]*pow(env.gamma, i) for i in range(config.n_steps)])
                q_table[s_t, a_t]+=config.alpha*(n_step_r+pow(env.gamma, config.n_steps)*q_table[state_next_idx, action_next_idx]-q_table[s_t, a_t])
                n_step_episode.pop(0)
                # 更新策略
                best_action_idx=np.argmax(q_table[s_t])
                policy[s_t]=np.full(env.n_actions, config.epsilon/env.n_actions)
                policy[s_t][best_action_idx]=1-config.epsilon+config.epsilon/env.n_actions
            
            state=state_next
            action=action_next
        
        # 对 episode 尾部剩余转移做截断 n-step 更新
        while len(n_step_episode) > 0:
            m = len(n_step_episode)
            n_step_r = np.sum(
                [n_step_episode[i][2] * pow(env.gamma, i) for i in range(m)]
            )
            state, action, reward = n_step_episode.pop(0)
            s_t = env.state_to_index(state)
            a_t = env.actions.index(action)
            q_table[s_t, a_t] += config.alpha * (
                n_step_r
                + pow(env.gamma, m) * q_table[state_next_idx, action_next_idx]
                - q_table[s_t, a_t]
            )
            # 更新策略
            best_action_idx=np.argmax(q_table[s_t])
            policy[s_t]=np.full(env.n_actions, config.epsilon/env.n_actions)
            policy[s_t][best_action_idx]=1-config.epsilon+config.epsilon/env.n_actions

        iterations+=1
        delta=np.max(np.abs(q_table-last_q_table))
        if delta<config.threshold:
            no_improvement_count+=1
            if no_improvement_count>=config.patience:
                converged=True
                #break
        else:
            no_improvement_count=0
        last_q_table=np.copy(q_table)

    return NStepSARSAResult(policy=policy,q_table=q_table,converged=converged,iterations=iterations)


if __name__=="__main__":
    env=GridWorld()
    config=NStepSARSAConfig()
    start_time=time.time()
    result=n_step_sarsa(env,config)

    elapsed_time=time.time()-start_time

    env=GridWorld(policy=result.policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value=env.get_true_value_by_policy(result.policy)
    env.render_with_state_value(state_value, title="n-step Sarsa state value", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    best_action_idx=np.argmax(result.policy,axis=1)
    policy=np.array([np.eye(env.n_actions)[best_action_idx[s_idx]] for s_idx in range(env.n_states)])
    env=GridWorld(policy=policy)
    state_value=env.get_true_value_by_policy(policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(state_value, title="n-step Sarsa state value", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()

    print("\n=== 运行配置与耗时 ===")
    print("算法: n-step Sarsa")
    print(f"配置参数: {asdict(config)}")
    print(f"迭代信息: iterations={result.iterations}, converged={result.converged}")
    print(f"算法运行时间: {elapsed_time:.6f} 秒")

    env_id=generate_env_id(env)
    if has_optimal_solution(env_id):
        optimal_solution = load_optimal_solution(env_id)
        true_state_value = optimal_solution.value
        result_value_sum = np.sum(state_value)
        true_value_sum = np.sum(true_state_value)
        error = np.mean(np.abs(true_state_value - state_value))
        
        print(f"总状态值: {result_value_sum:.5f}, 最优总状态值: {true_value_sum:.5f}")
        print(f"mean abs error: {error:.5f}")