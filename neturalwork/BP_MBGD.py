"""BP_MBGD.py
两层的前馈神经网络实现异或网络，使用小批量梯度下降法进行训练.
代码中符号保持与书中一致
d: 输入层神经元个数
q: 隐藏层神经元个数
l: 输出层神经元个数
eta: 学习率
E:累计误差
E_k:第k个样本的误差
v_ih:第i个输入神经元到第h个隐藏神经元的权重
w_hj:第h个隐藏神经元到第j个输出神经元的权重
gamma_h:第h个隐藏神经元的阈值
theta_j:第j个输出神经元的阈值
alpha:隐层神经元的输入
b: 隐层神经元的输出
beta: 输出层神经元的输入
y_hat:输出层神经元的输出
B: 批次大小
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 输入数据集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 输出数据集
y = np.array([[0], [1], [1], [0]])

#调整输入输出矩阵
X=X.T#(d,m)
Y=y.T#(l,m)

# 参数设置
d = 2# 输入层神经元个数
q = 3# 隐藏层神经元个数
l = 1# 输出层神经元个数
eta = 0.05# 学习率
max_iterations= 50000# 迭代次数
threshold = 0.001# 误差阈值
B = 2# 批次大小

class BP_MBGD:
    def __init__(self, d, q, l, eta, max_iter, threshold, B, seed):
        self.d = d
        self.q = q
        self.l = l
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold
        self.B = B

        # 权重与阈值
        np.random.seed(seed)
        self.v=np.random.rand(self.d,self.q)
        self.w=np.random.rand(self.q,self.l)
        self.gamma=np.random.rand(self.q,1)
        self.theta=np.random.rand(self.l,1)
    
    def forward(self, X):
        alpha=self.v.T @ X
        b=sigmoid(alpha-self.gamma)
        beta=self.w.T @ b
        Y_hat=sigmoid(beta-self.theta)
        return b, Y_hat
    
    def train(self, X, Y):
        m=X.shape[1]
        num_batches = m // self.B# 批次数量
        
        for count in range(self.max_iter):
            # 随机打乱样本顺序
            indices = np.random.permutation(m)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]
            
            # 按批次遍历
            for batch_idx in range(num_batches):
                # 获取当前批次数据
                start_idx = batch_idx * self.B
                end_idx = start_idx + self.B
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]
                
                b, Y_hat = self.forward(X_batch)
                
                g=(Y_batch-Y_hat)*sigmoid_derivative(Y_hat)
                e=(self.w @ g)*sigmoid_derivative(b)

                delta_w=self.eta*(b@g.T)/self.B
                delta_v=self.eta*(X_batch@e.T)/self.B
                delta_theta=-self.eta*(np.sum(g, axis=1, keepdims=True))/self.B
                delta_gamma=-self.eta*(np.sum(e, axis=1, keepdims=True))/self.B
                
                self.w+=delta_w
                self.v+=delta_v
                self.theta+=delta_theta
                self.gamma+=delta_gamma
            
            # 累计误差
            if (count+1) % 1000 == 0:
                _, Y_hat_full = self.forward(X)
                E_k = 0.5 * np.sum((Y - Y_hat_full) ** 2)
                E = E_k / m
                
                if E < self.threshold:
                    print(f"\n在第 {count + 1} 次迭代时收敛！")
                    print(f"最终平均误差: {E:.6f}")
                    break
        
        print(f"\n达到最大迭代次数 {self.max_iter}")
        _, Y_hat_full = self.forward(X)
        E_k = 0.5 * np.sum((Y - Y_hat_full) ** 2)
        E = E_k / m
        print(f"最终平均误差: {E:.6f}")
    
    def predict(self, X):
        b, Y_hat = self.forward(X)
        return Y_hat

if __name__ == "__main__":
    bp_mbgd = BP_MBGD(d, q, l, eta, max_iterations, threshold, B, seed=0)
    bp_mbgd.train(X, Y)
    y_hat = bp_mbgd.predict(X)
    print("输出结果为：",y_hat)