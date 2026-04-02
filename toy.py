import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(a):
    """
    输入：sigmoid 的输出 a
    输出：a * (1 - a)
    """
    return a * (1.0 - a)


class BP_SGD:
    """
    标准 BP 算法（随机梯度下降）
    两层前馈网络：输入层 — 隐藏层 — 输出层
    """
    def __init__(self, d, q, l, eta, max_iter, threshold):
        self.d = d
        self.q = q
        self.l = l
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold

        # 权重与阈值（小随机数初始化，更稳定）
        self.v = np.random.randn(d, q) * 0.01      # (d, q)
        self.w = np.random.randn(q, l) * 0.01      # (q, l)
        self.gamma = np.zeros((q, 1))              # 隐层阈值
        self.theta = np.zeros((l, 1))              # 输出层阈值

    def forward(self, x):
        """
        前向传播
        x: (d, 1)
        """
        alpha = self.v.T @ x                       # (q, 1)
        b = sigmoid(alpha - self.gamma)            # 隐层输出
        beta = self.w.T @ b                        # (l, 1)
        y_hat = sigmoid(beta - self.theta)         # 输出层输出
        return b, y_hat

    def train(self, X, y):
        """
        X: (N, d)
        y: (N, l)
        """
        N = X.shape[0]

        for it in range(self.max_iter):
            # ---- 随机选择一个样本 ----
            k = np.random.randint(N)
            x = X[k].reshape(-1, 1)                # (d, 1)
            y_k = y[k].reshape(-1, 1)              # (l, 1)

            # ---- 前向传播 ----
            b, y_hat = self.forward(x)

            # ---- 反向传播 ----
            # 输出层梯度 g (l, 1)
            g = y_hat * (1 - y_hat) * (y_k - y_hat)

            # 隐层梯度 e (q, 1)
            e = b * (1 - b) * (self.w @ g)

            # ---- 参数更新 ----
            self.w += self.eta * (b @ g.T)
            self.v += self.eta * (x @ e.T)
            self.theta -= self.eta * g
            self.gamma -= self.eta * e

            # ---- 累计误差（周期性检查即可）----
            if it % 1000 == 0:
                E = 0.0
                for i in range(N):
                    _, y_pred = self.forward(X[i].reshape(-1, 1))
                    E += 0.5 * np.sum((y[i].reshape(-1, 1) - y_pred) ** 2)

                if E < self.threshold:
                    print(f"在第 {it} 次迭代提前收敛，累计误差 E={E:.6f}")
                    break

        print("训练结束")
        return self

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            _, y_hat = self.forward(X[i].reshape(-1, 1))
            preds.append(y_hat)
        return preds

# 输入数据集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 输出数据集
y = np.array([[0], [1], [1], [0]])

# 参数设置
d = 2# 输入层神经元个数
q = 3# 隐藏层神经元个数
l = 1# 输出层神经元个数
eta = 0.05# 学习率
max_iterations= 100000# 迭代次数
threshold = 0.01# 误差阈值

if __name__ == "__main__":
    bp_sgd=BP_SGD(d, q, l, eta, max_iterations, threshold)
    bp_sgd.train(X, y)
    print("预测结果：", bp_sgd.predict(X))
