"""
任意层数的前馈神经网络实现（改进版）
修复了权重初始化、反向传播等问题
"""

import numpy as np

def sigmoid(x):
    # 防止溢出
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """sigmoid 函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

class BPNetworkImproved:
    def __init__(self, layer_sizes, eta=0.05, max_iter=10000, threshold=0.01):
        """
        初始化 BP 神经网络
        参数:
            layer_sizes: list, 每层神经元数量，如 [2, 3, 1] 表示 2 输入，3 隐层，1 输出
            eta: float, 学习率
            max_iter: int, 最大迭代次数
            threshold: float, 误差阈值
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)  # 总层数（包括输入层和输出层）
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold
        
        self.weights = []  # W
        self.biases = []   # b
        
        # 使用 Xavier 初始化方法初始化权重
        for i in range(self.n_layers - 1):
            # Xavier 初始化：权重从均匀分布 U[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))] 中采样
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入数据，形状为 (input_dim, n_samples)
        返回:
            A: 每层的加权输入列表
            Z: 每层的激活输出列表
        """
        A = []
        Z = []
        Z.append(x)  # Z[0] = X
        
        for l in range(self.n_layers - 1):
            # 计算加权输入: A_l = W_l^T @ Z_{l-1} + b_l
            A_l = self.weights[l].T @ Z[l] + self.biases[l]
            # 计算激活输出: Z_l = sigmoid(A_l)
            Z_l = sigmoid(A_l)
            A.append(A_l)
            Z.append(Z_l)

        return A, Z
    
    def compute_loss(self, Y_pred, Y_true):
        """计算均方误差损失"""
        m = Y_true.shape[1]
        loss = 0.5 * np.sum((Y_pred - Y_true) ** 2) / m
        return loss
    
    def train(self, X, Y):
        """
        训练神经网络
        参数:
            X: 训练数据，形状为 (input_dim, n_samples)
            Y: 标签数据，形状为 (output_dim, n_samples)
        """
        m = X.shape[1]
        
        for count in range(self.max_iter):
            # 前向传播
            A, Z = self.forward(X)
            
            # 计算损失
            if (count + 1) % 100 == 0:
                loss = self.compute_loss(Z[-1], Y)
                print(f"Iteration {count + 1}/{self.max_iter}, Loss: {loss:.6f}")
                
                # 检查是否收敛
                if loss < self.threshold:
                    print(f"\n在第 {count + 1} 次迭代时收敛！")
                    print(f"最终损失: {loss:.6f}")
                    break
            
            # 反向传播
            deltas = [None] * (self.n_layers - 1)
            
            # 计算输出层的误差项
            # delta_L = (Y_pred - Y_true) * sigmoid'(A_L)
            output_error = Z[-1] - Y
            output_activation_derivative = sigmoid_derivative(A[-1])
            deltas[-1] = output_error * output_activation_derivative
            
            # 反向传播计算隐藏层的误差项
            for l in range(self.n_layers - 3, -1, -1):
                # delta_l = (W_{l+1} @ delta_{l+1}) * sigmoid'(A_l)
                weighted_error = self.weights[l + 1] @ deltas[l + 1]
                activation_derivative = sigmoid_derivative(A[l])
                deltas[l] = weighted_error * activation_derivative
            
            # 更新权重和偏置
            for l in range(self.n_layers - 1):
                # 梯度: dW_l = Z_{l-1} @ delta_l^T / m
                # 注意：对于第一层，Z[l-1] 应该是 Z[l] 的前一个激活值
                if l == 0:
                    z_prev = X
                else:
                    z_prev = Z[l]
                
                # 权重更新
                grad_w = z_prev @ deltas[l].T / m
                self.weights[l] -= self.eta * grad_w
                
                # 偏置更新
                grad_b = np.sum(deltas[l], axis=1, keepdims=True) / m
                self.biases[l] -= self.eta * grad_b
        
        print("训练完成")

    def predict(self, x):
        """
        预测
        参数:
            x: 输入数据
        返回:
            预测输出
        """
        A, Z = self.forward(x)
        return Z[-1]


if __name__ == "__main__":
    # 异或问题为例
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    X = X.T
    Y = Y.T

    layer_sizes = [2, 3, 1]

    bp_network = BPNetworkImproved(layer_sizes, eta=0.5, max_iter=10000)

    bp_network.train(X, Y)

    print("\n预测结果:")
    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")
