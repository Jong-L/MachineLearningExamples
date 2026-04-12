"""
使用Softmax激活函数进行多分类的示例
"""

import numpy as np
from BP_multilayer_network_improved import BPNetwork


def create_iris_like_dataset():
    """创建类似鸢尾花的简化三分类数据集"""
    np.random.seed(42)
    
    # 类别0: 特征值较小
    class_0 = np.random.randn(20, 2) * 0.5 + np.array([1, 1])
    # 类别1: 特征值中等
    class_1 = np.random.randn(20, 2) * 0.5 + np.array([3, 3])
    # 类别2: 特征值较大
    class_2 = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
    
    X = np.vstack([class_0, class_1, class_2]).T
    
    # One-hot编码标签
    Y = np.zeros((3, 60))
    Y[0, :20] = 1   # 前20个样本属于类别0
    Y[1, 20:40] = 1 # 中间20个样本属于类别1
    Y[2, 40:] = 1   # 后20个样本属于类别2
    
    return X, Y


def main():
    print("=" * 60)
    print("Softmax多分类示例")
    print("=" * 60)
    
    # 创建数据集
    X, Y = create_iris_like_dataset()
    print(f"\n数据集大小: {X.shape[1]} 个样本, {X.shape[0]} 个特征")
    print(f"标签形状: {Y.shape}")
    print(f"类别分布: 每类20个样本")
    
    # 创建神经网络
    # 2个输入特征 -> 10个隐藏神经元 -> 3个输出（对应3个类别）
    network = BPNetwork(
        layer_sizes=[2, 10, 3],
        eta=0.1,
        max_iter=5000,
        threshold=0.01,
        activation='tanh',
        softmax_output=True  # 启用Softmax
    )
    
    print("\n开始训练...")
    print("-" * 60)
    network.train(X, Y)
    
    # 预测
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    
    predictions = network.predict(X)
    predicted_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(Y, axis=0)
    
    # 计算准确率
    accuracy = np.mean(predicted_classes == true_classes) * 100
    
    print(f"\n总体准确率: {accuracy:.2f}%")
    print(f"正确分类: {np.sum(predicted_classes == true_classes)}/{len(true_classes)}")
    
    # 显示部分预测结果
    print("\n前10个样本的预测详情:")
    print("-" * 60)
    for i in range(min(10, X.shape[1])):
        true_class = true_classes[i]
        pred_class = predicted_classes[i]
        probs = predictions[:, i]
        status = "✓" if true_class == pred_class else "✗"
        
        print(f"样本 {i+1:2d}: 真实={true_class}, 预测={pred_class} {status}")
        print(f"         概率分布: [类别0: {probs[0]:.4f}, 类别1: {probs[1]:.4f}, 类别2: {probs[2]:.4f}]")
    
    # 测试新样本
    print("\n" + "=" * 60)
    print("测试新样本")
    print("=" * 60)
    
    test_samples = np.array([
        [1.0, 1.2],   # 应该属于类别0
        [3.1, 2.9],   # 应该属于类别1
        [5.0, 5.1],   # 应该属于类别2
    ]).T
    
    test_predictions = network.predict(test_samples)
    test_classes = np.argmax(test_predictions, axis=0)
    
    for i in range(test_samples.shape[1]):
        probs = test_predictions[:, i]
        print(f"\n测试样本 {i+1}: 特征={test_samples[:, i]}")
        print(f"  预测类别: {test_classes[i]}")
        print(f"  概率分布: [类别0: {probs[0]:.4f}, 类别1: {probs[1]:.4f}, 类别2: {probs[2]:.4f}]")


if __name__ == "__main__":
    main()
