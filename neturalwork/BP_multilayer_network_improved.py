"""
任意层数的前馈神经网络实现 - 支持可替换的激活函数
"""

import numpy as np


class ActivationFunction:
    """激活函数基类"""
    def forward(self, x):
        """前向传播"""
        raise NotImplementedError
    
    def backward(self, x):
        """反向传播（计算导数）"""
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    """Sigmoid激活函数"""
    def forward(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)

class ReLU(ActivationFunction):
    """ReLU激活函数"""
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)

class Tanh(ActivationFunction):
    """Tanh激活函数"""
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        t = np.tanh(x)
        return 1 - t ** 2

class LeakyReLU(ActivationFunction):
    """Leaky ReLU激活函数"""
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)

class Softmax(ActivationFunction):
    """Softmax激活函数"""
    def forward(self, x):
        # 数值稳定性处理：减去最大值
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def backward(self, x):
        # Softmax的导数计算较为复杂，通常在损失函数中直接计算
        # 这里返回一个占位值，实际使用时配合交叉熵损失
        s = self.forward(x)
        return s * (1 - s)

class Linear(ActivationFunction):
    """线性激活函数（恒等映射），适用于回归问题输出层"""
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)

def get_activation_function(name):
    """根据名称获取激活函数实例"""
    activations = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'leaky_relu': LeakyReLU(),
        'softmax': Softmax(),
        'linear': Linear()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"不支持的激活函数: {name}. 支持的激活函数: {list(activations.keys())}")
    
    return activations[name.lower()]


class BPNetwork:
    def __init__(self, layer_sizes, eta=0.05, max_iter=15000, threshold=0.005, 
                 activation='sigmoid', output_activation=None, softmax_output=False):
        """
        初始化 BP 神经网络
        参数:
            layer_sizes: list, 每层神经元数量，如 [2, 3, 1] 表示 2 输入，3 隐层，1 输出
            eta: float, 学习率
            max_iter: int, 最大迭代次数
            threshold: float, 误差阈值
            activation: str or ActivationFunction, 隐藏层激活函数
            output_activation: str or ActivationFunction, 输出层激活函数（默认与隐藏层相同）
            softmax_output: bool, 是否在输出层使用Softmax（用于多分类问题）
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)  # 总层数（包括输入层和输出层）
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold
        self.softmax_output = softmax_output
        
        # 设置激活函数
        if isinstance(activation, str):
            self.activation_func = get_activation_function(activation)
        elif isinstance(activation, ActivationFunction):
            self.activation_func = activation
        else:
            raise TypeError("activation必须是字符串或ActivationFunction实例")
        
        # 如果指定了softmax_output，则输出层强制使用Softmax
        if softmax_output:
            self.output_activation_func = Softmax()
        elif output_activation is None:
            self.output_activation_func = self.activation_func
        elif isinstance(output_activation, str):
            self.output_activation_func = get_activation_function(output_activation)
        elif isinstance(output_activation, ActivationFunction):
            self.output_activation_func = output_activation
        else:
            raise TypeError("output_activation必须是字符串或ActivationFunction实例")
        
        self.weights = []  # W
        self.biases = []   # b
        
        for i in range(self.n_layers - 1):
            # 根据激活函数选择合适的权重初始化方法
            if isinstance(self.activation_func, (ReLU, LeakyReLU)):
                # He初始化（适合ReLU系列）
                limit = np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier初始化（适合Sigmoid/Tanh）
                limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        A = []
        Z = []
        Z.append(x)  # Z[0]=X
        
        for l in range(self.n_layers - 1):
            A_l = self.weights[l].T @ Z[l]
            
            # 选择适当的激活函数
            if l == self.n_layers - 2:  # 最后一层（输出层）
                Z_l = self.output_activation_func.forward(A_l - self.biases[l])
            else:  # 隐藏层
                Z_l = self.activation_func.forward(A_l - self.biases[l])
                
            A.append(A_l)
            Z.append(Z_l)

        return A, Z
    
    def _compute_delta(self, A, Z, Y, l, Delta, m):
        """计算第 l 层的 delta（梯度），供 update_batch 和 train 复用"""
        if l == self.n_layers - 2:  # 输出层
            if self.softmax_output:
                delta_l = (Z[-1] - Y) / m
            else:
                # 对于线性输出或其他激活函数
                if isinstance(self.output_activation_func, Linear):
                    delta_l = (Z[-1] - Y) / m
                elif isinstance(self.output_activation_func, Sigmoid):
                    delta_l = (Z[-1] - Y) * self.output_activation_func.backward(A[l] - self.biases[l]) / m
                elif isinstance(self.output_activation_func, ReLU):
                    delta_l = (Z[-1] - Y) * self.output_activation_func.backward(A[l] - self.biases[l]) / m
                elif isinstance(self.output_activation_func, Tanh):
                    delta_l = (Z[-1] - Y) * self.output_activation_func.backward(A[l] - self.biases[l]) / m
                elif isinstance(self.output_activation_func, LeakyReLU):
                    delta_l = (Z[-1] - Y) * self.output_activation_func.backward(A[l] - self.biases[l]) / m
                else:
                    # 通用情况
                    delta_l = (Z[-1] - Y) * self.output_activation_func.backward(A[l] - self.biases[l]) / m
        else:  # 隐藏层
            if isinstance(self.activation_func, Sigmoid):
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
            elif isinstance(self.activation_func, ReLU):
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
            elif isinstance(self.activation_func, Tanh):
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
            elif isinstance(self.activation_func, LeakyReLU):
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
            elif isinstance(self.activation_func, Linear):
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
            else:
                # 通用情况
                delta_l = (self.weights[l+1] @ Delta[l+1]) * self.activation_func.backward(A[l] - self.biases[l])
        return delta_l

    def update_batch(self, X, Y):
        """
        执行单步批量更新（前向传播 + 反向传播 + 参数更新）
        适用于 DQN 等需要逐步训练的强化学习算法
        
        参数:
            X: numpy array, shape=(input_dim, batch_size)
            Y: numpy array, shape=(output_dim, batch_size)
        返回:
            loss: float
        """
        m = X.shape[1]
        
        # 前向传播
        A, Z = self.forward(X)
        
        # 计算损失
        loss = self.compute_loss(Z[-1], Y)
        
        # 反向传播
        Delta = [None] * (self.n_layers - 1)
        for l in range(self.n_layers - 2, -1, -1):
            delta_l = self._compute_delta(A, Z, Y, l, Delta, m)
            
            # 更新权重和偏置
            self.weights[l] -= self.eta * Z[l] @ delta_l.T
            self.biases[l] += self.eta * np.sum(delta_l, axis=1, keepdims=True)
            Delta[l] = delta_l
        
        return loss

    def train(self, X, Y):
        m = X.shape[1]
        for count in range(self.max_iter):
            loss = self.update_batch(X, Y)
            
            if (count + 1) % 500 == 0:
                print(f"Iteration {count+1}, Loss: {loss}")
                if loss < self.threshold:
                    print(f"在第 {count+1} 次迭代时收敛！")
                    print(f"最终误差为：{loss}")
                    break
        
        if count == self.max_iter - 1:
            print(f"达到最大迭代次数，最终误差为：{loss}")

    def predict(self, x):
        A, Z = self.forward(x)
        return Z[-1]
    
    def copy_parameters_from(self, other):
        """从另一个 BPNetwork 实例深拷贝参数，用于 target network 同步"""
        if self.layer_sizes != other.layer_sizes:
            raise ValueError(f"网络结构不匹配：{self.layer_sizes} vs {other.layer_sizes}")
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]
    
    def compute_loss(self, Y_pred, Y):
        """计算损失函数"""
        if self.softmax_output:
            # 交叉熵损失（用于多分类）
            # 添加小的epsilon避免log(0)
            epsilon = 1e-8
            Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(Y * np.log(Y_pred_clipped), axis=0))
        else:
            # 均方误差（用于回归或二分类）
            return np.mean(np.square(Y_pred - Y)) / 2


if __name__ == "__main__":
    # 异或问题为例
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    X = X.T
    Y = Y.T

    print("=== 使用Sigmoid激活函数 ===")
    layer_sizes = [2, 3, 1]
    bp_network_sigmoid = BPNetwork(layer_sizes, eta=0.1, max_iter=50000, activation='sigmoid')
    bp_network_sigmoid.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network_sigmoid.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")

    print("\n=== 使用ReLU激活函数 ===")
    bp_network_relu = BPNetwork(layer_sizes, eta=0.01, max_iter=50000, activation='relu')
    bp_network_relu.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network_relu.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")

    print("\n=== 使用Tanh激活函数 ===")
    bp_network_tanh = BPNetwork(layer_sizes, eta=0.1, max_iter=50000, activation='tanh')
    bp_network_tanh.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network_tanh.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")

    print("\n=== 使用Leaky ReLU激活函数 ===")
    bp_network_leaky_relu = BPNetwork(layer_sizes, eta=0.01, max_iter=50000, activation='leaky_relu')
    bp_network_leaky_relu.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network_leaky_relu.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")

    # 测试Softmax - 使用简单的三分类问题
    print("\n=== 使用Softmax激活函数（三分类问题） ===")
    # 创建一个简单的三分类数据集
    X_multi = np.array([
        [0.1, 0.2, 0.15],   # 类别 0
        [0.9, 0.8, 0.85],   # 类别 1
        [0.5, 0.5, 0.5],    # 类别 2
        [0.15, 0.25, 0.2],  # 类别 0
        [0.85, 0.75, 0.8],  # 类别 1
        [0.45, 0.55, 0.5]   # 类别 2
    ]).T
    
    # One-hot编码的标签
    Y_multi = np.array([
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
        [0, 0, 1],  # 类别 2
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
        [0, 0, 1]   # 类别 2
    ]).T
    
    layer_sizes_multi = [3, 5, 3]  # 3输入，5隐藏，3输出（3个类别）
    bp_network_softmax = BPNetwork(
        layer_sizes_multi, 
        eta=0.1, 
        max_iter=10000, 
        activation='tanh',
        softmax_output=True  # 启用Softmax输出
    )
    bp_network_softmax.train(X_multi, Y_multi)

    print("\n预测结果：")
    for i in range(X_multi.shape[1]):
        input_val = X_multi[:, i]
        predicted = bp_network_softmax.predict(input_val.reshape(-1, 1))
        true_class = np.argmax(Y_multi[:, i])
        pred_class = np.argmax(predicted)
        print(f"样本 {i+1}: 真实类别={true_class}, 预测类别={pred_class}, 概率分布={predicted.flatten()}")