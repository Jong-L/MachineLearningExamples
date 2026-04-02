"""BP_SGD.py
两层的前馈神经网络实现异或网络，使用随机梯度下降法进行训练，即周志华《机器学习》中所谓的标准BP算法
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

# 参数设置
d = 2# 输入层神经元个数
q = 3# 隐藏层神经元个数
l = 1# 输出层神经元个数
eta = 0.05# 学习率
max_iterations= 100000# 迭代次数
threshold = 0.01# 误差阈值

class BP_SGD:
    def __init__(self, d, q, l, eta, max_iter, threshold):
        self.d = d
        self.q = q
        self.l = l
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold

        # 权重与阈值
        self.v=np.random.rand(self.d,self.q)
        self.w=np.random.rand(self.q,self.l)
        self.gamma=np.random.rand(self.q,1)
        self.theta=np.random.rand(self.l,1)           
    
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

    def train(self,X,y):
        for count in range(self.max_iter):
            k=np.random.randint(0,X.shape[0])
            x_k=X[k].reshape(-1,1)#转换为列向量
            y_k=y[k].reshape(-1,1)
            #计算\hat{y}
            b, y_hat = self.forward(x_k)

            #g shape(l,1)
            g=(y_k-y_hat)*sigmoid_derivative(y_hat)
            #e shape(q,1)
            e=(self.w@g)*sigmoid_derivative(b)
            #更新量
            delta_w=self.eta*(b@g.T)
            delta_v=self.eta*(x_k@e.T)
            delta_theta=-self.eta*g
            delta_gamma=-self.eta*e

            #更新
            self.w+=delta_w
            self.v+=delta_v
            self.theta+=delta_theta
            self.gamma+=delta_gamma

            #计算累计误差
            result_y_hat=[]
            E=0
            if (count+1) % 1000 == 0:#每1000次迭代打印一次误差
                for k in range(X.shape[0]):
                    x_k=X[k].reshape(-1,1)
                    y_k=y[k].reshape(-1,1)
                    b, y_hat = self.forward(x_k)
                    result_y_hat.append(y_hat)
                    E_k=0.5*(y_k-y_hat)**2
                    E+=E_k
                if E < self.threshold:
                    print(f"训练完成，迭代次数为：{count}")
                    break
        
        if count == self.max_iter-1:
            print("达到最大迭代次数，最终误差为：",E)
        
        print("最终预测结果为：",result_y_hat)

    def bp_network(self,x):
        return self.forward(x)

if __name__ == "__main__":
    bp_sgd = BP_SGD(d, q, l, eta, max_iterations, threshold)
    bp_sgd.train(X, y)
    b,y_hat=bp_sgd.bp_network(np.array([[0], [0]]))#测试
    print("测试结果为：",y_hat)

