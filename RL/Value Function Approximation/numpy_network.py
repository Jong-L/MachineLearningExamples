"""
使用 NumPy 实现的简单神经网络（用于 DQN 当没有 PyTorch 时）
"""

import numpy as np


class NeuralNetwork:
    """使用 NumPy 实现的简单神经网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3
    
    def backward(self, X, y_pred, y_true, learning_rate):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差
        delta3 = (y_pred - y_true) / m
        dW3 = self.a2.T @ delta3
        db3 = np.sum(delta3, axis=0, keepdims=True)
        
        # 隐藏层 2 误差
        delta2 = (delta3 @ self.W3.T) * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # 隐藏层 1 误差
        delta1 = (delta2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # 更新权重
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def copy_weights_from(self, other):
        """从另一个网络复制权重"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()
