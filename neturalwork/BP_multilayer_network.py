"""BP_Network.py：AI编写
通用的多层前馈神经网络实现
支持任意数量的隐藏层，使用随机梯度下降法 (SGD) 进行训练

网络结构可配置，例如：
- [2, 3, 1] 表示输入层 2 个神经元，隐藏层 3 个，输出层 1 个
- [2, 4, 3, 1] 表示输入层 2 个，第一个隐藏层 4 个，第二个隐藏层 3 个，输出层 1 个

符号说明（与周志华《机器学习》保持一致）：
- layer_sizes: 各层神经元数量列表
- weights: 权重列表，weights[i] 是第 i 层到第 i+1 层的权重矩阵
- biases: 阈值/偏置列表，biases[i] 是第 i+1 层的阈值
- activations: 各层的激活值（输出）
- deltas: 各层的误差项（用于反向传播）
- eta: 学习率
"""

import numpy as np

def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    """
    Sigmoid 函数的导数
    注意：这里直接传入 sigmoid 的输出值，而不是原始输入
    σ'(x) = σ(x)(1 - σ(x))
    """
    return output * (1 - output)

class BPNetwork:
    def __init__(self, layer_sizes, eta=0.05, max_iter=100000, threshold=0.01):
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
        
        # 初始化权重和阈值（偏置）
        # weights[i] 是第 i 层到第 i+1 层的权重矩阵，形状为 (layer_sizes[i], layer_sizes[i+1])
        # biases[i] 是第 i+1 层的阈值，形状为 (layer_sizes[i+1], 1)
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            # 使用小的随机值初始化权重和偏置
            w = np.random.rand(layer_sizes[i], layer_sizes[i+1])
            b = np.random.rand(layer_sizes[i+1], 1)
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """
        前向传播算法
        
        参数:
            x: 输入向量，形状为 (d, 1)，其中 d 是输入层神经元个数
        
        返回:
            activations: 列表，包含所有层的激活值（包括输入层和输出层）
                        activations[0] 是输入层，activations[-1] 是输出层
        """
        activations = [x]  # 保存每一层的输出，第 0 层是输入
        
        # 逐层计算前向传播
        for i in range(self.n_layers - 1):
            # 计算当前层的输入：alpha = W^T * a_prev - b
            z = self.weights[i].T @ activations[-1] - self.biases[i]
            # 应用激活函数得到当前层的输出
            a = sigmoid(z)
            activations.append(a)
        
        return activations
    
    def backward(self, activations, y_true):
        """
        反向传播算法（计算梯度）
        
        参数:
            activations: 前向传播得到的各层激活值列表
            y_true: 真实标签，形状为 (l, 1)，其中 l 是输出层神经元个数
        
        返回:
            weight_grads: 权重的梯度列表
            bias_grads: 阈值的梯度列表
        """
        # 初始化梯度列表
        weight_grads = []
        bias_grads = []
        
        # 计算输出层的误差项 delta
        # delta = (y_true - y_hat) * sigmoid_derivative(y_hat)
        # 注意：根据记忆中的经验，阈值更新符号应为正，所以这里保持 y_true - y_hat
        output_error = y_true - activations[-1]
        output_delta = output_error * sigmoid_derivative(activations[-1])
        
        # 将输出层误差项加入列表（从后往前）
        deltas = [output_delta]
        
        # 反向传播误差到隐藏层
        # delta_h = (W_h @ delta_next) * sigmoid_derivative(a_h)
        for i in range(self.n_layers - 2, 0, -1):
            # 计算第 i 层的误差项
            error = self.weights[i] @ deltas[0]
            delta = error * sigmoid_derivative(activations[i])
            deltas.insert(0, delta)  # 插入到列表开头
        
        # 计算梯度
        # grad_W = eta * (a_prev @ delta.T)
        # grad_b = -eta * delta （注意负号，因为阈值更新是 theta += delta_theta）
        for i in range(self.n_layers - 1):
            weight_grad = activations[i] @ deltas[i].T
            bias_grad = -deltas[i]
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads, bias_grads):
        """
        更新权重和阈值
        
        参数:
            weight_grads: 权重的梯度列表
            bias_grads: 阈值的梯度列表
        """
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * weight_grads[i]
            self.biases[i] += self.eta * bias_grads[i]
    
    def compute_loss(self, X, y):
        """
        计算累计误差
        
        参数:
            X: 输入数据集，形状为 (n_samples, n_features)
            y: 输出数据集，形状为 (n_samples, n_output)
        
        返回:
            E: 累计误差
        """
        E = 0
        predictions = []
        
        for k in range(X.shape[0]):
            x_k = X[k].reshape(-1, 1)
            y_k = y[k].reshape(-1, 1)
            
            activations = self.forward(x_k)
            y_hat = activations[-1]
            predictions.append(y_hat)
            
            # 均方误差的一半
            E_k = 0.5 * np.sum((y_k - y_hat) ** 2)
            E += E_k
        
        return E, predictions
    
    def train(self, X, y, verbose=True):
        """
        训练神经网络（标准 BP 算法/SGB）
        
        参数:
            X: 输入数据集，形状为 (n_samples, n_features)
            y: 输出数据集，形状为 (n_samples, n_output)
            verbose: 是否打印训练信息
        """
        for count in range(self.max_iter):
            # 随机选择一个样本（SGD）
            k = np.random.randint(0, X.shape[0])
            x_k = X[k].reshape(-1, 1)
            y_k = y[k].reshape(-1, 1)
            
            # 前向传播
            activations = self.forward(x_k)
            
            # 反向传播计算梯度
            weight_grads, bias_grads = self.backward(activations, y_k)
            
            # 更新参数
            self.update_parameters(weight_grads, bias_grads)
            
            # 定期检查和打印误差
            if verbose and (count + 1) % 1000 == 0:
                E, predictions = self.compute_loss(X, y)
                
                if verbose:
                    print(f"迭代次数：{count + 1}, 累计误差：{E:.6f}")
                
                # 检查是否达到误差阈值
                if E < self.threshold:
                    print(f"\n训练完成！迭代次数：{count + 1}, 最终误差：{E:.6f}")
                    break
        else:
            # 如果达到最大迭代次数
            E, predictions = self.compute_loss(X, y)
            print(f"\n达到最大迭代次数 {self.max_iter}, 最终误差：{E:.6f}")
        
        return predictions
    
    def predict(self, x):
        """
        预测单个样本
        
        参数:
            x: 输入向量，形状为 (d, 1) 或 (d,)
        
        返回:
            输出预测值
        """
        x = x.reshape(-1, 1)
        activations = self.forward(x)
        return activations[-1]
    
    def predict_batch(self, X):
        """
        批量预测
        
        参数:
            X: 输入数据集，形状为 (n_samples, n_features)
        
        返回:
            predictions: 预测结果列表
        """
        predictions = []
        for k in range(X.shape[0]):
            x_k = X[k].reshape(-1, 1)
            y_hat = self.predict(x_k)
            predictions.append(y_hat)
        return predictions


if __name__ == "__main__":
    # 示例：异或问题
    print("=" * 60)
    print("两层神经网络（一个隐藏层）解决异或问题")
    print("=" * 60)
    
    # 输入数据集
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # 输出数据集
    y = np.array([[0], [1], [1], [0]])
    
    # 创建网络：2 输入，3 隐藏层，1 输出
    network_2layer = BPNetwork(layer_sizes=[2, 3, 1], eta=0.05, max_iter=100000, threshold=0.01)
    
    # 训练
    predictions = network_2layer.train(X, y)
    
    # 测试
    print("\n测试结果:")
    for i in range(4):
        test_input = X[i].reshape(-1, 1)
        output = network_2layer.predict(test_input)
        print(f"输入：{test_input.T} -> 输出：{output[0, 0]:.6f} (期望：{y[i, 0]})")
    
    print("\n" + "=" * 60)
    print("三层神经网络（两个隐藏层）解决异或问题")
    print("=" * 60)
    
    # 创建更深的网络：2 输入，4 隐藏层，3 隐藏层，1 输出
    network_3layer = BPNetwork(layer_sizes=[2, 4, 3, 1], eta=0.05, max_iter=100000, threshold=0.01)
    
    # 训练
    predictions = network_3layer.train(X, y)
    
    # 测试
    print("\n测试结果:")
    for i in range(4):
        test_input = X[i].reshape(-1, 1)
        output = network_3layer.predict(test_input)
        print(f"输入：{test_input.T} -> 输出：{output[0, 0]:.6f} (期望：{y[i, 0]})")
    
    print("\n" + "=" * 60)
    print("对比说明：通过修改 layer_sizes 参数，可以轻松改变网络层数！")
    print("=" * 60)
