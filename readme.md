# Machine Learning 机器学习算法实现项目

> **AI分析并编写**

本项目是一个综合性的机器学习算法实现仓库，涵盖了从基础线性模型到深度强化学习的多种经典算法。所有算法均采用 Python 实现，部分实现基于 NumPy 从零开始构建，以深入理解算法原理。

## 项目结构

```
machine learning/
├── linear model/           # 线性模型
├── neturalwork/            # 神经网络
├── Decision Tree/          # 决策树
├── RL/                     # 强化学习
├── SVM/                    # 支持向量机
├── examples with frames/   # 框架使用示例
└── dataset/                # 数据集
```

## 模块详解

### 1. 线性模型 (linear model)

包含经典的线性分类与回归算法实现：

- **LDA (线性判别分析)**：`LDA.py` - 从零实现的线性判别分析
- **LDA with sklearn**：`LDA with sklearn.py` - 使用 sklearn 的 LDA 实现
- **逻辑回归**：`logistic regression2.py` - 包含特征映射的逻辑回归实现，支持非线性决策边界
- **Iris 分类示例**：`#Iris.py` - 基于 Iris 数据集的分类实践

### 2. 神经网络 (neturalwork)

完整的反向传播神经网络实现，包含多种优化策略：

| 文件 | 说明 |
|------|------|
| `BP_BGD.py` | 批量梯度下降实现 |
| `BP_SGD.py` | 随机梯度下降实现 |
| `BP_MBGD.py` | 小批量梯度下降实现 |
| `BP_multilayer_network.py` | 任意层数前馈神经网络，支持自定义网络结构 |
| `BP_multilayer_network_improved.py` | 改进版 BP 网络，包含优化策略 |
| `test_mnist.py` / `test_mnist_improved.py` | MNIST 手写数字识别测试 |
| `example_softmax.py` | Softmax 回归示例 |
| `BP_Derivation.md` | 反向传播算法的数学推导文档 |

**核心特性**：
- 支持任意层数的网络结构配置
- 多种梯度下降优化策略
- MNIST 数据集测试验证

### 3. 决策树 (Decision Tree)

- **`no sklearn.py`**：从零实现的决策树算法，使用 Car Evaluation 数据集进行车辆可接受性评估
- 实现了信息增益计算、树的构建与预测
- 数据集包含 1728 个样本，6 个属性特征

### 4. 强化学习 (RL)

本项目最丰富的模块，实现了完整的强化学习算法体系：

#### 4.1 环境定义
- **`grid_world.py`**：网格世界环境，支持自定义地图、禁止区域、目标位置
  - 5x5 默认网格
  - 可视化渲染（状态值、策略）
  - 确定性状态转移模型

#### 4.2 贝尔曼方程 (BellmanEquation)
- **`policy_evaluation.py`**：策略评估算法

#### 4.3 贝尔曼最优方程 (BellamOptimalityEquation)
- **`value_iteration.py`**：值迭代算法
- **`policy_iteration.py`**：策略迭代算法
- **`truncated_policy_iteration.py`**：截断策略迭代

#### 4.4 蒙特卡洛方法 (Monte Carlo Learning)
- **`MCBasic.py`**：基础蒙特卡洛
- **`MCExploringStarts.py`**：探索起点蒙特卡洛
- **`MC_epsilon_greedy.py`**：ε-贪婪策略蒙特卡洛

#### 4.5 时序差分学习 (TD learning)
- **`sarsa.py`**：SARSA 算法（同策略）
- **`q_learning_version1.py` / `q_learning_version2.py`**：Q-Learning 算法（异策略）
- **`expected_sarsa.py`**：Expected SARSA
- **`n_step_sarsa.py`**：n步 SARSA
- **`sarsa_glie.py`**：GLIE 策略 SARSA
- **`TD_state_value.py`**：TD 状态值估计

#### 4.6 值函数逼近 (Value Function Approximation)
- **`DQN_numpy.py`**：基于 NumPy 的 DQN 实现
- **`DQN_pytorch.py`**：基于 PyTorch 的 DQN 实现
- **`TD-Linear_state_value.py`**：线性函数逼近状态值
- **`sarsa_with_linear_function_approximation.py`**：带线性逼近的 SARSA
- **`numpy_network.py`**：NumPy 神经网络工具

#### 4.7 工具类
- **`optimal_solution_manager.py`**：最优解管理器，支持保存/加载最优策略
- **`optimal_solutions/`**：存储预计算的最优解

### 5. 框架使用示例 (examples with frames)

- **`linearRegression.py`**：使用 sklearn 的加州房价预测示例
  - 线性回归、Lasso 回归对比
  - 数据标准化、训练测试集划分
  - 模型评估指标（R²、RMSE）
  
- **`Text Classification with Naive Bayes.py`**：朴素贝叶斯文本分类

### 6. 数据集 (dataset)

- **`car+evaluation/`**：汽车评估数据集（决策树使用）
- **`ex2data1.txt` / `ex2data2.txt`**：逻辑回归测试数据
- **`iris.data.csv`**：鸢尾花数据集
- **`watermelon3.0 alpha.xlsx`**：西瓜数据集

## 快速开始

### 环境要求
- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- scikit-learn (部分示例)
- PyTorch (DQN 实现)

### 运行示例

```bash
# 运行神经网络示例
cd neturalwork
python BP_multilayer_network.py

# 运行强化学习示例
cd RL/BellamOptimalityEquation
python value_iteration.py

# 运行决策树
cd Decision\ Tree
python no\ sklearn.py
```

## 学习路径建议

1. **入门**：从 `linear model/` 和 `examples with frames/` 开始，理解基础机器学习概念
2. **进阶**：学习 `neturalwork/` 中的反向传播实现，理解神经网络原理
3. **深入**：探索 `RL/` 模块，从值迭代、策略迭代到 Q-Learning、DQN
4. **实践**：使用 `Decision Tree/` 和自定义数据集进行实践

## 特点与亮点

- **从零实现**：大部分算法基于 NumPy 实现，便于理解底层原理
- **完整注释**：代码包含详细的中文注释和数学公式说明
- **可视化支持**：强化学习模块支持策略和状态值的可视化
- **模块化设计**：清晰的模块划分，便于学习和扩展
- **理论与实践结合**：既有算法实现，也有实际数据集测试

## 参考资料

- 《机器学习》（周志华）
- 《深度学习》（Goodfellow et al.）
- 《Reinforcement Learning: An Introduction》（Sutton & Barto）
- 赵世钰老师的强化学习课程

---

*本项目用于学习和研究目的，持续更新中。*
