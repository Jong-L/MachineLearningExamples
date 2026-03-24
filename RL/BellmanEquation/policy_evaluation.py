"""
其实贝尔曼公式相关内容更多在grid_world.py中，这里只是简单地展示了不同策略下状态值是多少
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld


def policy_evaluation(env:GridWorld, gamma: float = 0.9) -> np.ndarray:
    v=env.get_true_value_by_policy()
    # 创建包含两个子图的画布，1行2列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 在第一个子图中绘制状态价值
    env.render_with_state_value(v, ax=ax1)
    
    # 在第二个子图中绘制策略
    env.render_with_policy(ax=ax2)
    ax2.set_title('Policy')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    env=GridWorld()#均匀分布策略情况
    #policy_evaluation(env)

    #随机初始化策略
    policy_random=np.random.rand(25,5)
    policy_random=policy_random/policy_random.sum(axis=1,keepdims=True)
    policy_random[0]=np.array([0,0,1,0,0])#用于观察状态0
    env=GridWorld(policy=policy_random)
    policy_evaluation(env)
