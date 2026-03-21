import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld

class ValueIteration:
    @staticmethod
    def q_value(env: GridWorld,V:np.ndarray, s: int, a: Tuple[int,int]) -> float:
        state= env.index_to_state(s)
        next_state, reward = env.step(state, a)
        next_state_idx=env.state_to_index(next_state)
        q= reward + env.gamma * V[next_state_idx]
        return q

    @staticmethod
    def iterate( env: GridWorld, threshold: float = 1e-6,max_iter: int = 1000) ->Tuple[np.ndarray, np.ndarray]:
        V= np.zeros(env.n_states)
        for count in range(max_iter):
            new_V= np.zeros(env.n_states)
            best_actions = np.zeros(env.n_states, dtype=int) 
            for s in range(env.n_states):
                max_q= -np.inf
                max_a_idx=0
                for a_idx,a in enumerate(env.actions):
                    q=ValueIteration.q_value(env, V,s, a)
                    if q>max_q:
                        max_q= q
                        max_a_idx= a_idx
                new_V[s]= max_q
                best_actions[s]= max_a_idx

            delta=np.max(np.abs(V - new_V))
            V=new_V
            if delta < threshold:
                print(f"Converged at iteration {count + 1}")
                break
        if count == max_iter - 1:
            print("Reached maximum iterations without converging.")
        policy= ValueIteration.extract_policy(env, best_actions)
        return V,policy
    
    @staticmethod
    def extract_policy(env: GridWorld, best_actions: np.ndarray) -> np.ndarray:
        policy = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            policy[s] = np.eye(env.n_actions)[best_actions[s]]
        return policy

if __name__ == "__main__":
    env = GridWorld()
    V, policy = ValueIteration.iterate(env)
    #policy[0]=np.array([0,0,0,0,0])
    env= GridWorld(policy=policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    env.render_with_state_value(title="true state value with policy", ax=ax1)
    env.render_with_policy(ax=ax2)
    plt.show()