"""
对比原始版和改进版 BP 神经网络在 digits 数据集上的表现
"""

import numpy as np
import sys
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from neturalwork.BP_multilayer_network import BPNetwork
from neturalwork.BP_multilayer_network_improved import BPNetworkImproved


def load_and_prepare_data():
    """加载并准备数据"""
    print("正在加载 digits 数据集...")
    
    digits = load_digits()
    X, Y = digits.data, digits.target
    X = X / 16.0
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    # 转置数据
    X_train_T = X_train.T
    X_test_T = X_test.T
    
    # One-hot 编码
    lb = LabelBinarizer()
    Y_train_bin = lb.fit_transform(Y_train).T
    Y_test_bin = lb.transform(Y_test).T
    
    return X_train_T, X_test_T, Y_train_bin, Y_test_bin, lb


def evaluate_predictions(Y_pred, Y_true, lb):
    """评估预测结果"""
    pred_labels = lb.inverse_transform(Y_pred.T)
    true_labels = lb.inverse_transform(Y_true.T)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy


def test_original_version(X_train, X_test, Y_train, Y_test, lb):
    """测试原始版本"""
    print("\n" + "=" * 60)
    print("测试原始版 BP 神经网络")
    print("=" * 60)
    
    layer_sizes = [64, 50, 10]
    bp_net = BPNetwork(layer_sizes=layer_sizes, eta=0.1, max_iter=1000)
    
    print(f"\n网络结构: {layer_sizes}")
    print("开始训练...")
    start_time = time.time()
    bp_net.train(X_train, Y_train)
    end_time = time.time()
    
    print(f"训练耗时: {end_time - start_time:.2f} 秒")
    
    # 评估
    Y_train_pred = bp_net.predict(X_train)
    train_accuracy = evaluate_predictions(Y_train_pred, Y_train, lb)
    print(f"训练集准确率: {train_accuracy * 100:.2f}%")
    
    Y_test_pred = bp_net.predict(X_test)
    test_accuracy = evaluate_predictions(Y_test_pred, Y_test, lb)
    print(f"测试集准确率: {test_accuracy * 100:.2f}%")
    
    return train_accuracy, test_accuracy


def test_improved_version(X_train, X_test, Y_train, Y_test, lb):
    """测试改进版本"""
    print("\n" + "=" * 60)
    print("测试改进版 BP 神经网络")
    print("=" * 60)
    
    layer_sizes = [64, 50, 10]
    bp_net = BPNetworkImproved(
        layer_sizes=layer_sizes,
        eta=0.5,
        max_iter=1000,
        threshold=0.001
    )
    
    print(f"\n网络结构: {layer_sizes}")
    print("开始训练...")
    start_time = time.time()
    bp_net.train(X_train, Y_train)
    end_time = time.time()
    
    print(f"训练耗时: {end_time - start_time:.2f} 秒")
    
    # 评估
    Y_train_pred = bp_net.predict(X_train)
    train_accuracy = evaluate_predictions(Y_train_pred, Y_train, lb)
    print(f"训练集准确率: {train_accuracy * 100:.2f}%")
    
    Y_test_pred = bp_net.predict(X_test)
    test_accuracy = evaluate_predictions(Y_test_pred, Y_test, lb)
    print(f"测试集准确率: {test_accuracy * 100:.2f}%")
    
    return train_accuracy, test_accuracy


def main():
    """主函数"""
    print("=" * 60)
    print("BP 神经网络版本对比测试")
    print("=" * 60)
    
    # 加载数据
    X_train, X_test, Y_train, Y_test, lb = load_and_prepare_data()
    
    print(f"\n数据集信息:")
    print(f"  训练集大小: {X_train.shape[1]}")
    print(f"  测试集大小: {X_test.shape[1]}")
    print(f"  输入维度: {X_train.shape[0]}")
    print(f"  类别数: {Y_train.shape[0]}")
    
    # 测试原始版本
    orig_train_acc, orig_test_acc = test_original_version(
        X_train, X_test, Y_train, Y_test, lb
    )
    
    # 测试改进版本
    impr_train_acc, impr_test_acc = test_improved_version(
        X_train, X_test, Y_train, Y_test, lb
    )
    
    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果总结")
    print("=" * 60)
    print(f"\n{'指标':<20} {'原始版':>10} {'改进版':>10} {'提升':>10}")
    print("-" * 60)
    print(f"{'训练集准确率':<18} {orig_train_acc*100:>9.2f}% {impr_train_acc*100:>9.2f}% {((impr_train_acc-orig_train_acc)*100):>+9.2f}%")
    print(f"{'测试集准确率':<18} {orig_test_acc*100:>9.2f}% {impr_test_acc*100:>9.2f}% {((impr_test_acc-orig_test_acc)*100):>+9.2f}%")
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
