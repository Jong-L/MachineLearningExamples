import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, gamma=0.9):
        self.rows = 5
        self.cols = 5
        self.n_states = self.rows * self.cols
        self.gamma = gamma
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.action_prob = 1.0 / len(self.actions)

        # Special cells from the figure, using 0-based coordinates.
        self.forbidden = {
            (1, 1), (1, 2),
            (2, 2),
            (3, 1), (3, 3),
            (4, 1),
        }
        self.target = (3, 2)

        self.r_boundary = -1.0
        self.r_forbidden = -1.0
        self.r_target = 1.0

    def state_to_index(self, state):
        r, c = state
        return r * self.cols + c

    def index_to_state(self, index):
        return divmod(index, self.cols)

    def is_inside(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, state, action):

        nr = state[0] + action[0] #  新行坐标 = 当前行坐标 + 动作的行变化
        nc = state[1] + action[1] #  新列坐标 = 当前列坐标 + 动作的列变化
        next_state = (nr, nc)

        if not self.is_inside(next_state):
            return state, self.r_boundary

        if next_state in self.forbidden:
            return state, self.r_forbidden

        if next_state == self.target:
            return next_state, self.r_target

        return next_state, 0.0

    def sample_next(self, state, rng):
        action_id = rng.integers(len(self.actions))
        return self.step(state, self.actions[action_id])

    def build_linear_system(self):
        p_pi = np.zeros((self.n_states, self.n_states))#  状态转移矩阵
        r_pi = np.zeros(self.n_states)#  状态奖励向量

        for s in range(self.n_states):
            state = self.index_to_state(s)
            for action in self.actions:
                next_state, reward = self.step(state, action)
                ns = self.state_to_index(next_state)
                p_pi[s, ns] += self.action_prob#在边界可能会多个动作转移到同一个状态，所以用+=
                r_pi[s] += self.action_prob * reward
        return p_pi, r_pi

    def true_value(self):
        p_pi, r_pi = self.build_linear_system()
        a = np.eye(self.n_states) - self.gamma * p_pi
        v = np.linalg.solve(a, r_pi)
        return v.reshape(self.rows, self.cols) #  将解得的扁平化状态价值向量重塑为原始网格形状


def build_features(env):
    features = np.zeros((env.n_states, 7))
    for s in range(env.n_states):
        r, c = env.index_to_state(s)
        rn = r / (env.rows - 1)
        cn = c / (env.cols - 1)
        is_target = 1.0 if (r, c) == env.target else 0.0
        near_forbidden = 0.0
        for dr, dc in env.actions:
            neighbor = (r + dr, c + dc)
            if neighbor in env.forbidden:
                near_forbidden = 1.0
                break

        features[s] = np.array([
            1.0,
            rn,
            cn,
            rn * cn,
            rn ** 2,
            cn ** 2,
            is_target + near_forbidden,
        ])
    return features


def td_linear_policy_evaluation(
    env,
    features,
    true_v,
    alpha=0.05,
    num_steps=30000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    theta = np.zeros(features.shape[1])
    errors = []

    valid_starts = [
        env.state_to_index((r, c))
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) != env.target
    ]
    state_idx = rng.choice(valid_starts)

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

        if next_state == env.target:
            state_idx = rng.choice(valid_starts)
        else:
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
    true_v = env.true_value()
    features = build_features(env)

    theta, errors = td_linear_policy_evaluation(
        env=env,
        features=features,
        true_v=true_v.reshape(-1),
        alpha=0.03,
        num_steps=40000,
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
