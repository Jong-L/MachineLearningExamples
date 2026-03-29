"""
网格地图，api设计仅仅部分参考了gym，实际上为了方便这个已经不是单纯的环境，是环境+策略+其他各种东西的混合物。
默认5*5，策略为均匀分布，进入边界时状态不变，可以进入禁止区域，进入目标状态后不终止
这是一个确定性模型,该模型以及在此基础上的强化学习算法都是按照确定性模型来写的，因此简化了部分代码
在该模型中，状态值v是一个一维向量，shape为(n_states,1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Set, Tuple
from typing import Optional

class GridWorld:
    def __init__(self, rows: int = 5, cols: int = 5, gamma= 0.9, policy: Optional[np.ndarray] = None) -> None:
        """
        policy: 策略矩阵，shape为(n_states, n_actions),policy[i,j]=p(a_j|s_i)
        """
        self.rows = rows
        self.cols = cols
        self.n_states = self.rows * self.cols
        self.gamma = gamma
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]#依次为左，右，上，下，不动
        self.n_actions = len(self.actions)

        if policy is None:
            self.policy = np.ones((self.n_states, len(self.actions))) / len(self.actions)
        else:
            self.policy = policy
        
        self.set_forbidden()
        self.set_target()
        self.set_rewards()

    def set_target(self, target_pos:Tuple[int,int]=(3, 2))-> None:
        self.target = target_pos

    def set_forbidden(self, forbidden_pos: Set[Tuple[int, int]] = None)-> Set[Tuple[int, int]]:
        if forbidden_pos is None:
            forbidden_pos = {
                (1, 1), (1, 2),
                (2, 2),
                (3, 1), (3, 3),
                (4, 1),
            }
        self.forbidden = forbidden_pos

    def set_rewards(self, r_boundary:float=-1.0, r_forbidden:float=-1.0, r_target:float=1.0)-> None:
        self.r_boundary = r_boundary
        self.r_forbidden = r_forbidden
        self.r_target = r_target
    
    def state_to_index(self, state):
        r, c = state
        return r * self.cols + c

    def index_to_state(self, index):
        return divmod(index, self.cols)

    def is_inside(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, state: Tuple[int, int], action: Tuple[int, int], stay_on_forbidden: bool = True) -> Tuple[Tuple[int, int], float]:
        """执行动作，返回下一个状态和奖励,如果是随机模型可以调整为返回下一个状态和奖励的概率分布,p(r|s,a)和p(s'|s,a)"""
        nr = state[0] + action[0]
        nc = state[1] + action[1]
        next_state = (nr, nc)

        if not self.is_inside(next_state):
            return state, self.r_boundary

        if next_state in self.forbidden:
            if stay_on_forbidden:
                return next_state, self.r_forbidden
            else:
                return state, self.r_forbidden

        if next_state == self.target:
            return next_state, self.r_target

        return next_state, 0.0

    def sample_next(self, state:Tuple[int,int], policy, rng: np.random.RandomState):#根据策略生成动作并执行
        s_idx = self.state_to_index(state)
        if policy is None:
            policy = self.policy
        action_probs = policy[s_idx]
        action_id = rng.choice(len(self.actions), p=action_probs)
        action = self.actions[action_id]
        next_state, reward = self.step(state, action)
        return (next_state,action, reward)

    def build_linear_system(self):#模型，p(s_j|s_i)=p[i,j]，r_i=r[i]
        p_pi = np.zeros((self.n_states, self.n_states))
        r_pi = np.zeros(self.n_states)

        for s in range(self.n_states):
            state = self.index_to_state(s)
            for action_id, action in enumerate(self.actions):
                next_state, reward = self.step(state, action)
                ns = self.state_to_index(next_state)
                p_pi[s, ns] += self.policy[s, action_id]#在边界多个动作会转移到一个状态，因此累加
                r_pi[s] += self.policy[s, action_id] * reward
        return p_pi, r_pi

    def true_value(self) -> np.ndarray:
        p_pi, r_pi = self.build_linear_system()
        a = np.eye(self.n_states) - self.gamma * p_pi
        v = np.linalg.solve(a, r_pi)
        return v

    def get_true_value_by_policy(self, policy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        通过求解线性方程组计算状态值
        
        policy: 可选的策略矩阵，shape为(n_states, n_actions)。如果不提供，使用self.policy
        """
        if policy is not None:
            # 临时使用传入的策略
            original_policy = self.policy
            self.policy = policy
            p_pi, r_pi = self.build_linear_system()
            self.policy = original_policy
        else:
            p_pi, r_pi = self.build_linear_system()
        
        a = np.eye(self.n_states) - self.gamma * p_pi
        v = np.linalg.solve(a, r_pi)
        return v

    def get_itrated_value_by_policy(
        self, 
        policy: Optional[np.ndarray] = None,
        threshold: float = 1e-6,
        max_iterations: int = 1000
    ) -> np.ndarray:
        """
        通过迭代法求解贝尔曼方程计算状态值
        
        policy: 可选的策略矩阵，shape为(n_states, n_actions)。如果不提供，使用self.policy
        threshold: 收敛阈值，当最大变化量小于该值时停止迭代
        max_iterations: 最大迭代次数
        
        返回：状态值向量 v, shape为(n_states,)
        """
        if policy is not None:
            # 临时使用传入的策略
            original_policy = self.policy
            self.policy = policy
            p_pi, r_pi = self.build_linear_system()
            self.policy = original_policy
        else:
            p_pi, r_pi = self.build_linear_system()
        
        # 初始化状态值为 0
        v = np.zeros(self.n_states)
        
        for iteration in range(max_iterations):
            # 贝尔曼备份：v = r + gamma * P * v
            v_new = r_pi + self.gamma * p_pi @ v
            
            # 检查是否收敛
            delta = np.max(np.abs(v_new - v))
            if delta < threshold:
                break
            
            v = v_new
        
        return v

    def value_vector_to_matrix(self, v: np.ndarray) -> np.ndarray:
        if v.shape != (self.n_states,):
            raise ValueError(f"`v` 的形状应为 {(self.n_states,)}，实际为 {v.shape}")
        return v.reshape(self.rows, self.cols)

    def render(self,ax:Optional[plt.Axes]=None):#绘制网格地图
        """
        ax:如果传入Axes对象，则作为子图绘制，否则绘制独立窗口
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.cols), max(4, self.rows)))
            show_polt=True
        else:
            show_polt=False


        # 让坐标系左上为(0,0)，更符合网格/矩阵习惯
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(self.cols) + 0.5)
        ax.set_yticks(np.arange(self.rows) + 0.5)
        ax.set_xticklabels(list(range(self.cols)))
        ax.set_yticklabels(list(range(self.rows)))
        ax.grid(False)

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.target:
                    facecolor = "tab:green"
                    label = "T"
                elif pos in self.forbidden:
                    facecolor = "lightgray"
                    label = "X"
                else:
                    facecolor = "white"
                    label = ""

                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax.add_patch(rect)

                if label:
                    ax.text(
                        c + 0.5, r + 0.5, label,
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="black",
                    )

        ax.set_title("Grid World Map (T=Target, X=Forbidden)")
        plt.tight_layout()
        if show_polt:
            plt.show()

    def render_with_state_value(
        self,
        v: Optional[np.ndarray] = None,
        title: str = "State Value V(s)",
        use_heatmap: bool = False,
        cmap: str = "coolwarm",
        ax:Optional[plt.Axes] = None,
    ):#绘制状态值数值图
        if v is None:
            v_mat = self.value_vector_to_matrix(self.get_true_value_by_policy())
        else:
            if v.shape == (self.n_states,):
                v_mat = self.value_vector_to_matrix(v)
            elif v.shape == (self.rows, self.cols):
                v_mat = v
            else:
                raise ValueError(
                    f"`v` 的形状应为 {(self.n_states,)} 或 {(self.rows, self.cols)}，实际为 {v.shape}"
                )

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.cols), max(4, self.rows)))
            show_plot = True
        else:
            show_plot = False
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(self.cols) + 0.5)
        ax.set_yticks(np.arange(self.rows) + 0.5)
        ax.set_xticklabels(list(range(self.cols)))
        ax.set_yticklabels(list(range(self.rows)))
        ax.grid(False)

        # 选择性地叠加热力图，但不影响“禁止/目标”区分标识
        if use_heatmap:
            im = ax.imshow(
                v_mat,
                origin="upper",
                cmap=cmap,
                extent=(0, self.cols, self.rows, 0),
                alpha=0.85,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)

                # 底色高亮：禁止格/目标格（默认不用热力图时用更强的底色）
                if not use_heatmap:
                    if pos == self.target:
                        bg = "tab:green"
                    elif pos in self.forbidden:
                        bg = "lightgray"
                    else:
                        bg = "white"
                    ax.add_patch(
                        patches.Rectangle((c, r), 1, 1, facecolor=bg, edgecolor="black", linewidth=1.2)
                    )
                else:
                    # 热力图模式下，用半透明底色确保可读性与区分性
                    if pos == self.target:
                        bg = "tab:green"
                    elif pos in self.forbidden:
                        bg = "lightgray"
                    else:
                        bg = None
                    if bg is not None:
                        ax.add_patch(
                            patches.Rectangle(
                                (c, r), 1, 1,
                                facecolor=bg, edgecolor="black", linewidth=1.0, alpha=0.35,
                            )
                        )

                # 数值标注：禁止格/目标格都显示 V(s) 数值
                v_text = f"{v_mat[r, c]:.2f}"
                text_color = "black"
                if use_heatmap and pos == self.target:
                    text_color = "black"
                ax.text(c + 0.5, r + 0.5, v_text, ha="center", va="center", fontsize=12, color=text_color)

                # 额外标识目标格（避免和数值重叠，放在角落）
                if pos == self.target:
                    ax.text(c + 0.12, r + 0.18, "T", ha="left", va="top", fontsize=10, fontweight="bold", color="black")

        ax.set_title(title + (" (heatmap)" if use_heatmap else ""))
        plt.tight_layout()
        if show_plot:
            plt.show()

    def render_with_policy(self,ax:Optional[plt.Axes] = None):#绘制策略图,箭头长度表示动作概率
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.cols), max(4, self.rows)))
            show_plot = True
        else:
            show_plot = False
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(self.cols) + 0.5)
        ax.set_yticks(np.arange(self.rows) + 0.5)
        ax.set_xticklabels(list(range(self.cols)))
        ax.set_yticklabels(list(range(self.rows)))
        ax.grid(False)

        # self.actions 内部表示为 (dr, dc): 行增量、列增量

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                s_idx = self.state_to_index(pos)
                
                # 绘制格子背景
                if pos == self.target:
                    facecolor = "tab:green"
                elif pos in self.forbidden:
                    facecolor = "lightgray"
                else:
                    facecolor = "white"
                    
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax.add_patch(rect)
                
                # 为每个动作绘制箭头
                for action_id, (dr, dc) in enumerate(self.actions):
                    prob = self.policy[s_idx, action_id]
                    if prob > 0.01:  # 只绘制概率大于阈值的动作
                        # 箭头长度与概率成正比，最大长度为0.4
                        arrow_length = 0.5 * prob
                        if action_id == 4:  # 原地不动用圆圈表示
                            circle = patches.Circle(
                                (c + 0.5, r + 0.5), 
                                0.1 * prob,
                                facecolor=	"#00EE00",
                                alpha=0.7,
                            )
                            ax.add_patch(circle)
                        else:
                            dx = dc * arrow_length
                            dy = dr * arrow_length
                            # 绘制箭头
                            ax.arrow(
                                c + 0.5, r + 0.5,  # 起点
                                dx, dy,  # 方向和长度
                                head_width=0.1 * prob,  # 箭头宽度与概率成正比
                                head_length=0.15 * prob,  # 箭头长度与概率成正比
                                fc="#00EE00",  # 箭头填充色
                                ec="#00EE00",  # 箭头边缘色
                                alpha=0.7,  # 透明度
                                length_includes_head=True  # 箭头长度包括箭头部分
                            )

        ax.set_title("Policy Visualization (Arrow length ∝ Action probability)")
        plt.tight_layout()
        if show_plot:
            plt.show()

if __name__ == "__main__":
    env = GridWorld()
    #env.render_with_policy()
    
    # 测试两种方法计算状态值
    print("Testing get_true_value_by_policy (linear system solution)...")
    v_linear = env.get_true_value_by_policy()
    print(f"Linear method result shape: {v_linear.shape}")
    print(f"Linear method max value: {np.max(v_linear):.4f}")
    print(f"Linear method min value: {np.min(v_linear):.4f}")
    
    print("\nTesting get_itrated_value_by_policy (iterative method)...")
    v_iterative = env.get_itrated_value_by_policy()
    print(f"Iterative method result shape: {v_iterative.shape}")
    print(f"Iterative method max value: {np.max(v_iterative):.4f}")
    print(f"Iterative method min value: {np.min(v_iterative):.4f}")
    
    print(f"\nDifference between two methods (max abs diff): {np.max(np.abs(v_linear - v_iterative)):.2e}")
    print("Both methods should produce very similar results!")
    
    # 测试传入不同策略的情况
    print("\n\nTesting with random policy...")
    policy_random = np.random.rand(25, 5)
    policy_random = policy_random / policy_random.sum(axis=1, keepdims=True)
    env_random = GridWorld(policy=policy_random)
    
    v_random_linear = env_random.get_true_value_by_policy()
    v_random_iterative = env_random.get_itrated_value_by_policy()
    
    print(f"Random policy - Linear method max value: {np.max(v_random_linear):.4f}")
    print(f"Random policy - Iterative method max value: {np.max(v_random_iterative):.4f}")
    print(f"Random policy - Difference: {np.max(np.abs(v_random_linear - v_random_iterative)):.2e}")