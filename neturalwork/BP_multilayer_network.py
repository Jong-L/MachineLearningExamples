"""
任意层数的前馈神经网络实现
建议阅读neturalwork\BP_Derivation.md
"""

import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    pass

class BPNetwork:
    def __init__(self, layer_sizes, eta=0.05, max_iter=15000, threshold=0.001):
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
        
        self.weights = []#W
        self.biases = []#b
        
        # self.biases.append(0)#输入层阈值填充，方便下标和公式对应,以后改
        # self.weights.append(0)
        np.random.seed(42)
        for i in range(self.n_layers-1):
            # 使用小的随机值初始化权重和偏置
            w = np.random.rand(layer_sizes[i], layer_sizes[i+1])*0.01
            b = np.random.rand(layer_sizes[i+1], 1)
            # limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            # w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            # b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        A=[]
        Z=[]
        Z.append(x)#Z[0]=X
        for l in range(self.n_layers-1):
            A_l=self.weights[l].T @ Z[l]
            Z_l=sigmoid(A_l-self.biases[l])
            A.append(A_l)
            Z.append(Z_l)

        return A,Z
    
    def train(self, X, Y):
        m = X.shape[1]
        for count in range(self.max_iter):
            A, Z = self.forward(X)

            if (count+1) % 500 == 0:
                E=self.compute_loss(Z[-1], Y)
                print(f"Iteration {count+1}, Loss: {E}")
                if E < self.threshold:
                    print(f"在第 {count+1} 次迭代时收敛！")
                    print(f"最终误差为：{E}")
                    break
            
            Delta = [None]*(self.n_layers-1)
            for l in range(self.n_layers-2, -1, -1):
                if l==self.n_layers-2:
                    delta_l=(Z[-1]-Y)*Z[-1]*(1-Z[-1])/m
                else:
                    delta_l = (self.weights[l+1] @ Delta[l+1])*Z[l+1]*(1-Z[l+1])

                # 更新权重和偏置
                self.weights[l] -= self.eta * Z[l] @ delta_l.T
                self.biases[l] += self.eta * np.sum(delta_l, axis=1, keepdims=True)
                Delta[l]=delta_l

        if count==self.max_iter-1:
            E=self.compute_loss(Z[-1], Y)
            print(f"达到最大迭代次数，最终误差为：{E}")


    def predict(self, x):
        A,Z=self.forward(x)
        return Z[-1]
    
    def compute_loss(self, Y_pred, Y):
        return np.mean(np.square(Y_pred - Y))/2

if __name__ == "__main__":
    #异或问题为例
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    X=X.T
    Y=Y.T

    layer_sizes = [2, 3, 1]

    bp_network=BPNetwork(layer_sizes, eta=0.1, max_iter=50000)

    bp_network.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")
