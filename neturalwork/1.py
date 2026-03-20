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
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 20000

# 初始化权重
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

for epoch in range(epochs):
    # 前向传播
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)
    
    # 计算误差
    error = y - predicted_output
    
    # 反向传播
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # 更新权重
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

print("最终预测结果:")
print(predicted_output)