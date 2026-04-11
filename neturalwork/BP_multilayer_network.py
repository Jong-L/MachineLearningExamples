"""
任意层数的前馈神经网络实现
"""

import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class BPNetwork:
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
        
        self.weights = []#W
        self.biases = []#b
        
        self.biases.append(0)#输入层阈值填充，方便下标和公式对应,以后改
        self.weights.append(0)
        for i in range(self.n_layers-1):
            # 使用小的随机值初始化权重和偏置
            w = np.random.rand(layer_sizes[i], layer_sizes[i+1])
            b = np.random.rand(layer_sizes[i+1], 1)
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        A=[]
        Z=[]
        Z.append(x)#Z[0]=X
        A.append(0)#填充
        for l in range(1,self.n_layers):
            A_l=self.weights[l].T @ Z[l-1]
            Z_l=sigmoid(A_l-self.biases[l])
            A.append(A_l)
            Z.append(Z_l)

        return A,Z
    
    
    def train(self, X, Y):
        m = X.shape[1]
        for count in range(self.max_iter):
            A, Z = self.forward(X)
            
            Delta = []
            for l in range(self.n_layers-1, 0, -1):
                if l==self.n_layers-1:
                    delta_l=(Z[-1]-Y)*Z[-1]*(1-Z[-1])/m
                else:
                    delta_l = (self.weights[l+1] @ Delta[0])*Z[l]*(1-Z[l])

                # 更新权重和偏置
                self.weights[l] -= self.eta * Z[l-1] @ delta_l.T
                self.biases[l] += self.eta * np.sum(delta_l, axis=1, keepdims=True)

                Delta.insert(0, delta_l)

        print("训练完成")


    def predict(self, x):
        A,Z=self.forward(x)
        return Z[-1]


if __name__ == "__main__":
    #异或问题为例
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    X=X.T
    Y=Y.T

    layer_sizes = [2, 3, 1]

    bp_network=BPNetwork(layer_sizes, eta=0.05, max_iter=50000)

    bp_network.train(X, Y)

    for i in range(4):
        input_val = X[:, i]
        predicted = bp_network.predict(input_val.reshape(-1, 1))
        true_val = Y[:, i]
        print(f"Input: {input_val}, Predicted Output: {predicted[0][0]:.4f}, True Output: {true_val[0]}")
