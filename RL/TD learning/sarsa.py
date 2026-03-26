"""
与TD(0)相比，SARSA是在估计q值，并依据q值来更新策略
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
class SARSAConfig:
    alpha: float = 0.003
    epsilon: float = 0.1#epsilon greedy policy
    n_episodes: int = 1000
    episode_length: int = 100
    seed: int = 42

# 生成episode
def episode_generate(env: GridWorld, state: Tuple[int, int], action: Tuple[int, int],policy: np.ndarray,episode_length: int,seed:int) -> List:
    episode = []
    #rng=np.random.default_rng(seed)
    rng=np.random.default_rng()
    next_state, reward = env.step(state, action)
    episode.append((state, action, reward))
    state=next_state
    for _ in range(episode_length):
        next_state,action,reward=env.sample_next(state,policy,rng)
        episode.append((state, action, reward))
        state = next_state
    
    return episode

def sarsa(env: GridWorld, config: SARSAConfig) -> np.ndarray:
    """返回策略矩阵"""
    rng=np.random.default_rng(config.seed)
    policy=env.policy#初始化策略
    q_table=np.zeros((env.n_states,env.n_actions))

    for _ in range(config.n_episodes):
        state_idx=rng.choice(env.n_states)
        state=env.index_to_state(state_idx)
        _,action,_=env.sample_next(state,policy,rng)
        action_idx=env.actions.index(action)
        episode=episode_generate(env,state,action,policy,config.episode_length,config.seed)

        for i, (state, action, reward) in enumerate(episode[:-1]):
            next_state, next_action, _ = episode[i + 1]
            next_state_idx = env.state_to_index(next_state)
            next_action_idx = env.actions.index(next_action)
            # 更新q值
            q_table[state_idx, action_idx] =q_table[state_idx, action_idx]+ config.alpha * (reward + env.gamma*q_table[next_state_idx, next_action_idx] - q_table[state_idx, action_idx])
            # 更新策略
            best_action_idx = np.argmax(q_table[state_idx])
            policy = np.full((env.n_states, env.n_actions), config.epsilon / env.n_actions)
            policy[np.arange(env.n_states), best_action_idx] = 1 - (env.n_actions - 1) * config.epsilon / env.n_actions
            #滑动到下一个状态
            state_idx = next_state_idx
            action_idx = next_action_idx
    
    return policy

if __name__ == "__main__":
    env = GridWorld()
    config = SARSAConfig()
    policy = sarsa(env, config)
    env=GridWorld(policy=policy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state_value=env.get_true_value_by_policy(policy)
    env.render_with_state_value(state_value, title="Sarsa V*(s)", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()


            
