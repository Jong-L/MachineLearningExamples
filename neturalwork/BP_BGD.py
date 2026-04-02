"""BP_BGD.py
两层的前馈神经网络实现异或网络，使用批量梯度下降法进行训练.
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

class BP_BGD:
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
