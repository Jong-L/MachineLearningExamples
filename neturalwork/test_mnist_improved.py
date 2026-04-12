"""
使用改进的 BP 神经网络（支持 Softmax）测试 MNIST 手写数字识别
"""

import numpy as np
import sys
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time

# 添加父目录到路径，以便导入改进版 BP 网络
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from neturalwork.BP_multilayer_network_improved import BPNetwork


def load_mnist_data(n_samples=None):
    """
    加载 MNIST 数据集
    
    参数:
        n_samples: 可选，使用的样本数量（用于快速测试）
    
    返回:
        X_train, X_test, Y_train, Y_test
    """
    print("正在加载 MNIST 数据集...")
    
    # 从 OpenML 加载 MNIST 数据集
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, Y = mnist.data, mnist.target.astype(int)
    
    # 归一化像素值到 [0, 1]
    X = X / 255.0
    
    if n_samples is not None:
        X = X[:n_samples]
        Y = Y[:n_samples]
        print(f"使用前 {n_samples} 个样本")
    
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"输入维度: {X_train.shape[1]}")
    
    return X_train, X_test, Y_train, Y_test


def prepare_data_for_bp(X_train, X_test, Y_train, Y_test):
    """
    将数据转换为 BP 神经网络所需的格式
    
    参数:
        X_train, X_test: 特征数据
        Y_train, Y_test: 标签数据
    
    返回:
        X_train_T, X_test_T, Y_train_bin_T, Y_test_bin_T, lb
    """
    # 转置数据以匹配神经网络的输入格式 (features, samples)
    X_train_T = X_train.T
    X_test_T = X_test.T
    
    # 将标签转换为 one-hot 编码
    lb = LabelBinarizer()
    Y_train_bin = lb.fit_transform(Y_train)
    Y_test_bin = lb.transform(Y_test)
    
    # 转置为 (classes, samples) 格式
    Y_train_bin_T = Y_train_bin.T
    Y_test_bin_T = Y_test_bin.T
    
    print(f"训练数据形状: {X_train_T.shape}")
    print(f"训练标签形状: {Y_train_bin_T.shape}")
    print(f"输出类别数: {Y_train_bin_T.shape[0]}")
    
    return X_train_T, X_test_T, Y_train_bin_T, Y_test_bin_T, lb


def evaluate_predictions(Y_pred, Y_true, lb):
    """
    评估预测结果
    
    参数:
        Y_pred: 预测的 one-hot 编码
        Y_true: 真实的 one-hot 编码
        lb: LabelBinarizer 对象
    
    返回:
        accuracy: 准确率
    """
    # 将 one-hot 编码转换回标签
    pred_labels = lb.inverse_transform(Y_pred.T)
    true_labels = lb.inverse_transform(Y_true.T)
    
    # 计算准确率
    accuracy = np.mean(pred_labels == true_labels)
    
    return accuracy


def main():
    """主函数"""
    # 配置参数
    N_SAMPLES = 5000  # 使用的样本数量
    HIDDEN_UNITS = 64  # 隐藏层神经元数量
    LEARNING_RATE = 1  # 学习率
    MAX_ITER = 2000  # 最大迭代次数
    ACTIVATION = 'tanh'  # 激活函数选择：'sigmoid', 'tanh', 'relu', 'leaky_relu'
    USE_SOFTMAX = True  # 是否使用 Softmax 输出层（推荐用于多分类）
    
    print(f"\n{'='*60}")
    print(f"改进版 BP 神经网络 - MNIST 手写数字识别")
    print(f"{'='*60}")
    print(f"\n配置参数:")
    print(f"  样本数量: {N_SAMPLES}")
    print(f"  隐藏层神经元: {HIDDEN_UNITS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  最大迭代次数: {MAX_ITER}")
    print(f"  激活函数: {ACTIVATION}")
    print(f"  使用 Softmax: {USE_SOFTMAX}")
    print()
    
    # 加载数据
    X_train, X_test, Y_train, Y_test = load_mnist_data(n_samples=N_SAMPLES)
    
    # 准备数据
    X_train_T, X_test_T, Y_train_bin_T, Y_test_bin_T, lb = prepare_data_for_bp(
        X_train, X_test, Y_train, Y_test
    )
    
    # 创建神经网络
    layer_sizes = [784, HIDDEN_UNITS, 10]
    
    print(f"\n网络结构: {layer_sizes}")
    print("正在创建神经网络...")
    
    bp_net = BPNetwork(
        layer_sizes=layer_sizes,
        eta=LEARNING_RATE,
        max_iter=MAX_ITER,
        threshold=0.01,  # 误差阈值
        activation=ACTIVATION,
        softmax_output=USE_SOFTMAX  # 启用 Softmax 输出层
    )
    
    # 训练网络
    print("\n开始训练...")
    start_time = time.time()
    bp_net.train(X_train_T, Y_train_bin_T)
    end_time = time.time()
    
    print(f"\n训练耗时: {end_time - start_time:.2f} 秒")
    
    # 在训练集上评估
    print("\n在训练集上评估...")
    Y_train_pred = bp_net.predict(X_train_T)
    train_accuracy = evaluate_predictions(Y_train_pred, Y_train_bin_T, lb)
    print(f"训练集准确率: {train_accuracy * 100:.2f}%")
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    Y_test_pred = bp_net.predict(X_test_T)
    test_accuracy = evaluate_predictions(Y_test_pred, Y_test_bin_T, lb)
    print(f"测试集准确率: {test_accuracy * 100:.2f}%")
    
    # 显示一些预测示例
    print("\n预测示例（前10个测试样本）:")
    print("-" * 60)
    n_examples = 10
    for i in range(n_examples):
        pred_label = np.argmax(Y_test_pred[:, i])
        true_label = np.argmax(Y_test_bin_T[:, i])
        match = "✓" if pred_label == true_label else "✗"
        
        # 如果使用了 Softmax，显示概率分布
        if USE_SOFTMAX:
            probs = Y_test_pred[:, i]
            prob_str = ", ".join([f"{p:.3f}" for p in probs])
            print(f"  样本 {i+1:2d}: 预测={pred_label}, 真实={true_label} {match}")
            print(f"           概率: [{prob_str}]")
        else:
            print(f"  样本 {i+1:2d}: 预测={pred_label}, 真实={true_label} {match}")
    
    # 统计每个数字的准确率
    print("\n各类别准确率统计:")
    print("-" * 60)
    pred_labels = lb.inverse_transform(Y_test_pred.T)
    true_labels = lb.inverse_transform(Y_test_bin_T.T)
    
    for digit in range(10):
        mask = true_labels == digit
        if np.sum(mask) > 0:
            digit_accuracy = np.mean(pred_labels[mask] == true_labels[mask])
            print(f"  数字 {digit}: 准确率 {digit_accuracy*100:5.1f}% "
                  f"({np.sum(pred_labels[mask] == digit)}/{np.sum(mask)})")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
