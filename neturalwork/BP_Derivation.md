[TOC]

> 前置知识：神经网络的基本想法，梯度下降优化方法。
>
> 个人文章不代表权威知识，如有错误请指出
>
> 代码示例在：[Jong-L/MachineLearningExamples](https://github.com/Jong-L/MachineLearningExamples)，其中的neturalwork目录下即为本文的所有代码。

对于任一随机样本，其样本的值$y$服从随机变量$Y$的概率分布，该样本输入到神经网络，输出为$\hat y$,相应的服从随机变量$\hat Y$的概率分布，神经网络的最终目的是最小化$\frac 12 \mathbb{E}[(\hat{Y}-Y)^2]$,假设对于每个样本出现概率完全相同（比如函数拟合问题），则最小化$\frac {1}{2m} \sum_{k=1}^m (\hat{y^k}-y^k)^2$。

假设我们希望在给定随机变量 $X$ 的一组随机样本 $\{x_i\}_{i=1}^n$ 的条件下，最小化目标函数
$$ J(w) = \mathbb{E}[f(w, X)] $$
求解该问题的**批量梯度下降（BGD）**、**小批量梯度下降（MBGD）**和**随机梯度下降（SGD）**算法分别为：

$$w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i), \quad \text{(BGD)}$$

$$w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j), \quad \text{(MBGD)}$$

$$ w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k). \quad \text{(SGD)} $$

我们将分别使用以上方法对网络的权重进行更新来构建神经网络。

# 两层前馈神经网络

## 网络结构

- 输入层：$d$ 个神经元
- 隐藏层：$q$ 个神经元，激活函数 $\sigma$，sigmoid函数
- 输出层：$l$ 个神经元，激活函数 $\sigma$

## 符号定义

| 符号 | 维度 | 含义 |
|------|------|------|
| $x_i^k$ | 标量 | 第 $k$个输入的第$i$个特征 |
| $v_{ih}$ | 标量 | 输入层第 $i$ 个神经元到隐藏层第 $h$ 个神经元的权重 |
| $\gamma_h$ | 标量 | 隐藏层第 $h$ 个神经元的阈值 |
| $\alpha_h$ | 标量 | 隐藏层第 $h$ 个神经元的输入 |
| $b_h$ | 标量 | 隐藏层第 $h$ 个神经元的输出 |
| $w_{hj}$ | 标量 | 隐藏层第 $h$ 个神经元到输出层第 $j$ 个神经元的权重 |
| $\theta_j$ | 标量 | 输出层第 $j$ 个神经元的阈值 |
| $\beta_j$ | 标量 | 输出层第 $j$ 个神经元的输入 |
| $\hat{y}_j^k$ | 标量 | 第$k$ 个样本输出层第 $j$ 个神经元的输出 |
| $y_j^k$ | 标量 | 第 $k$ 个输出的真实值的第j个特征 |

## SGD

### 逐元素形式

#### 前向传播

隐藏层第 $h$ 个神经元的输入：
$$\alpha_h = \sum_{i=1}^{d} v_{ih} x_i$$

隐藏层第 $h$ 个神经元的输出：
$$b_h = \sigma(\alpha_h - \gamma_h)$$

输出层第 $j$ 个神经元的输入：
$$\beta_j = \sum_{h=1}^{q} w_{hj} b_h$$

输出层第 $j$ 个神经元的输出：
$$\hat{y}_j = \sigma(\beta_j - \theta_j)$$

#### 误差函数

采用SGD，每次采用一个样本进行参数更新，对第$k$个样本，误差为：
$$E_k = \frac{1}{2} \sum_{j=1}^{l} (\hat{y}_j^k-y_j^k)^2$$

#### 反向传播

以 $w_{hj}$为例，误差函数对其梯度为：
$$
\frac{\partial E}{\partial w_{hj}}=\frac{\partial E}{\partial \hat{y}_j^k}\cdot \frac{\partial \hat{y}_j^k}{\partial \alpha_j^k}\cdot \frac{\partial \alpha_j^k}{\partial w_{hj}}=(\hat{y}_j^k-y_j^k)\frac{\partial \sigma(\alpha_j^k - \gamma_j)}{\partial \alpha_j^k}\cdot b_h
$$
sigmoid函数具有性质： $\frac{\partial \sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))$

于是 $\frac{\partial E}{\partial w_{hj}}=(\hat{y}_j^k-y_j^k)\hat{y}_j^k (1 - \hat{y}_j^k) b_h$

定义：

输出层梯度项：
$$g_j = \hat{y}_j^k (1 - \hat{y}_j^k) (y_j^k - \hat{y}_j^k)$$

隐藏层梯度项：
$$e_h = b_h (1 - b_h) \sum_{j=1}^{l} w_{hj} g_j$$



**参数更新**规则为：

$$\Delta w_{hj} = \eta g_j b_h$$

$$\Delta \theta_j = -\eta g_j$$

$$\Delta v_{ih} = \eta e_h x_i$$

$$\Delta \gamma_h = -\eta e_h$$



### 矩阵形式

#### 变量定义

| 矩阵/向量 | 代码变量 | 维度 | 含义 |
|-----------|----------|------|------|
| $\mathbf{x}$ | `x` | $(d, 1)$ | 输入向量 |
| $\mathbf{V}$ | `v` | $(d, q)$ | 输入层到隐藏层权重矩阵 |
| $\boldsymbol{\gamma}$ | `gamma` | $(q, 1)$ | 隐藏层阈值向量 |
| $\boldsymbol{\alpha}$ | `alpha` | $(q, 1)$ | 隐藏层输入 |
| $\mathbf{b}$ | `b` | $(q, 1)$ | 隐藏层输出 |
| $\mathbf{W}$ | `w` | $(q, l)$ | 隐藏层到输出层权重矩阵 |
| $\boldsymbol{\theta}$ | `theta` | $(l, 1)$ | 输出层阈值向量 |
| $\boldsymbol{\beta}$ | `beta` | $(l, 1)$ | 输出层输入 |
| $\hat{\mathbf{y}}$ | `y_hat` | $(l, 1)$ | 输出层输出 |
| $\mathbf{y}$ | `y_k` | $(l, 1)$ | 真实输出 |

#### 前向传播

$$\boldsymbol{\alpha} = \mathbf{V}^\top \mathbf{x}$$

$$\mathbf{b} = \sigma(\boldsymbol{\alpha} - \boldsymbol{\gamma})$$

$$\boldsymbol{\beta} = \mathbf{W}^\top \mathbf{b}$$

$$\hat{\mathbf{y}} = \sigma(\boldsymbol{\beta} - \boldsymbol{\theta})$$

#### 反向传播

输出层梯度：
$$\mathbf{g} = (\mathbf{y} - \hat{\mathbf{y}}) \odot \hat{\mathbf{y}} \odot (1 - \hat{\mathbf{y}})$$

隐藏层梯度：
$$\mathbf{e} = (\mathbf{W} \mathbf{g}) \odot \mathbf{b} \odot (1 - \mathbf{b})$$

#### 参数更新

$$\Delta \mathbf{W} = \eta \mathbf{b} \mathbf{g}^\top$$

$$\Delta \mathbf{V} = \eta \mathbf{x} \mathbf{e}^\top$$

$$\Delta \boldsymbol{\theta} = -\eta \mathbf{g}$$

$$\Delta \boldsymbol{\gamma} = -\eta \mathbf{e}$$



利用该网络拟合异或问题， 运行[BP_SGD.py](BP_SGD.py) ，结果如下：

![image-20260412182146954](D:\Cursor Program\machine learning\neturalwork\assets\image-20260412182146954.png)

这表明我们的网络构建是成功的。

## BGD

对应代码示例为 [BP_BGD.py](BP_BGD.py) 

每次使用所有样本，相应的误差函数为$\frac {1}{2m} \sum_{k=1}^m (\hat{y^k}-y^k)^2$，参数更新量为：

$$\Delta w_{hj} = \eta \frac 1m\sum_{k=1}^m g_j^k b_h^k$$

$$\Delta \theta_j = -\eta \frac 1m\sum_{k=1}^m g_j^k$$

$$\Delta v_{ih} = \eta \frac 1m\sum_{k=1}^m e_h^k x_i^k$$

$$\Delta \gamma_h = -\eta\frac 1m\sum_{k=1}^m e_h^k$$



同时将所有m个样本输入，输入矩阵$X_{d\times m}$,输出矩阵$Y_{l\times m}$。

分块(为了书写简便，从这里开始又用下标表示是第几个样本,需注意和上面SGD部分的区别）：
$$
\mathbf{X}=\begin{bmatrix}
 \mathbf x_1 & \mathbf x_2 & \cdots  &\mathbf x_m
\end{bmatrix},\mathbf{Y}=\begin{bmatrix}
 \mathbf y_1 & \mathbf y_2 & \cdots  &\mathbf y_m
\end{bmatrix}
$$
隐藏层输入：
$$
\boldsymbol{\alpha}_{q\times m}=\mathbf{V}^\top\mathbf {X}=\mathbf{X}=\begin{bmatrix}
 \mathbf{V}^\top\mathbf x_1 & \mathbf{V}^\top\mathbf x_2 & \cdots  &\mathbf{V}^\top\mathbf x_m
\end{bmatrix}
$$
隐藏层输出：
$$
\mathbf b_{q\times m}=\sigma(\boldsymbol{\alpha} - \boldsymbol{\gamma}_{q\times 1})
$$
利用numpy广播机制可直接在代码中书写以上公式。

输出层输入：
$$
\mathbf \beta_{l\times m}=\mathbf{W}^\top \mathbf{b}
$$
输出层输出：
$$
\mathbf Y_{l\times m}=\sigma(\boldsymbol{\beta} - \boldsymbol{\theta}_{l\times 1})
$$
输出层g：
$$
\mathbf g_{l\times m}=(\mathbf Y -\mathbf{\hat{Y}} )\odot \mathbf{\hat{Y}} \odot (1-\mathbf{\hat{Y}})
$$
y隐藏层e：
$$
\mathbf e_{q\times m}=(\mathbf{W} \mathbf{g}) \odot \mathbf{b} \odot (1 - \mathbf{b})
$$

#### 参数更新

$$\Delta \mathbf{W} = \frac {\eta}{m} \mathbf{b} \mathbf{g}^\top$$

$$\Delta \mathbf{V} = \frac {\eta}{m} \mathbf{X} \mathbf{e}^\top$$

$$\Delta \boldsymbol{\theta} = -\frac {\eta}{m} \sum_{i=1}^m\mathbf{g}_i$$

$$\Delta \boldsymbol{\gamma} = -\frac {\eta}{m}\sum_{i=1}^m\mathbf{e}_i$$



## MBGD

对应代码示例为： [BP_MBGD.py](BP_MBGD.py) 

BGD的代码实现可以比较轻易地改为MBGD，以B=2为例，每次从$X_{d\times m}$中随机选取两列组成新矩阵作为输入矩阵，相应的也从$Y_{l\times m}$得到输出矩阵，用它们更新一次网络的参数，然后选择剩下的两列，然后循环。可以把BGD当成$B=m$的MBGD。

# 任意层数的前馈神经网络（BGD）

对应代码示例为： [BP_multilayer_network.py](BP_multilayer_network.py) 

## 网络结构

设有 $L$ 层神经网络（包括输入层和输出层）：

- **第 0 层**：输入层，$n_0$ 个神经元
- **第 1 层到第 $L-1$ 层**：隐藏层，第 $l$ 层有 $n_l$ 个神经元
- **第 $L$ 层**：输出层，$n_L$ 个神经元

每层的激活函数为 $\sigma(\cdot)$。

## 符号说明

| 符号               | 维度 | 含义                                                       |
| ------------------ | ---- | ---------------------------------------------------------- |
| $L$                | 标量 | 网络总层数                                                 |
| $n_l$              | 标量 | 第 $l$ 层的神经元数量（$l = 0, 1, ..., L$）                |
| $m$                | 标量 | 训练样本总数（batch size）                                 |
| $x_i^{(k)}$        | 标量 | 第 $k$ 个样本的第 $i$ 个输入特征                           |
| $\alpha_j^{(l,k)}$ | 标量 | 第 $k$ 个样本在第 $l$ 层第 $j$ 个神经元的**加权输入**      |
| $z_j^{(l,k)}$      | 标量 | 第 $k$ 个样本在第 $l$ 层第 $j$ 个神经元的**激活输出**      |
| $w_{ij}^{(l)}$     | 标量 | 第 $l-1$ 层第 $i$ 个神经元到第 $l$ 层第 $j$ 个神经元的权重 |
| $b_j^{(l)}$        | 标量 | 第 $l$ 层第 $j$ 个神经元（阈值）                           |
| $\delta_j^{(l,k)}$ | 标量 | 第 $k$ 个样本在第 $l$ 层第 $j$ 个神经元的**误差项**        |
| $\eta$             | 标量 | 学习率（learning rate）                                    |
| $y_j^{(k)}$        | 标量 | 第 $k$ 个样本的输出层第 $j$ 个神经元的真实值               |
| $\hat{y}_j^{(k)}$  | 标量 | 第 $k$ 个样本的输出层第 $j$ 个神经元的预测值               |

## 矩阵/向量形式符号

| 矩阵/向量                   | 代码变量   | 维度               | 含义                  |
| --------------------------- | ---------- | ------------------ | --------------------- |
| $\mathbf{X}$                | `X`        | $(n_0, m)$         | 输入矩阵              |
| $\mathbf{Y}$                | `Y`        | $(n_L, m)$         | 真实输出矩阵          |
| $\mathbf{A}^{(l)}$          | `A[l]`     | $(n_l, m)$         | 第 $l$ 层加权输入矩阵 |
| $\mathbf{Z}^{(l)}$          | `Z[l]`     | $(n_l, m)$         | 第 $l$ 层激活输出矩阵 |
| $\mathbf{W}^{(l)}$          | `W[l]`     | $( n_{l-1},n_{l})$ | 第 $l$ 层权重矩阵     |
| $\mathbf{b}^{(l)}$          | `b[l]`     | $(n_l, 1)$         | 第 $l$ 层阈值向量     |
| $\boldsymbol{\Delta}^{(l)}$ | `Delta[l]` | $(n_l, m)$         | 第 $l$ 层误差项矩阵   |
| $\hat{\mathbf{Y}}$          | `Y_hat`    | $(n_L, m)$         | 预测输出矩阵          |

## 向前传播

对于第k个样本，第$l$​层第$j$​个神经元的输入为
$$
\alpha_j^{(l,k)}=\sum_{i=1}^{n_{l-1}}w_{ij}^{l}z_i^{(l-1,k)}
$$
当 $l = 1$ 时，$z_i^{(0,k)} = x_i^{(k)}$.

该神经元的输出为
$$
z_j^{(l,k)}=\sigma(\alpha_j^{(l,k)}-b_j^{(l)})
$$
**矩阵形式**：
$$
\mathbf{A}^{(l)}=(\mathbf{W}^{(l)})^T\mathbf{Z}^{(l-1)}\\
\mathbf{Z}^{(l)}=\sigma(\mathbf{A}^{(l)}-\mathbf b^{(l)})
$$

## 向后传播

最小化的目标函数为$E=\frac{1}{2m}\sum_{k=1}^m\sum_{j=1}^{n_l} (\hat{y}_j^k-y_j^k)^2$,为了能够迭代地求解梯度，对于第$l$层，定义 $\delta_p^{(l,k)}=\frac{\partial E}{\partial \alpha_p^{(l,k)}}$,称其为误差项，那么：
$$
\frac{\partial E}{\partial w_{pq}^l}=\frac{\partial E}{\partial \alpha_q^{(l,k)}}\cdot \frac{\partial \alpha_q^{(l,k)}}{\partial w_{pq}^l}=\delta_q^{(l,k)}\frac{\partial \alpha_q^{(l,k)}}{\partial w_{pq}^l}=\delta_q^{(l,k)}z_p^{(l-1,k)}\\
\frac{\partial E}{\partial b_p^{(l)}}=\frac{\partial E}{\partial z_p^{(l,k)}}\cdot \frac{\partial z_p^{(l,k)}}{\partial b_p^{(l)}}=\sum_{j=1}^{n_{l+1}}\delta_j^{(l+1,k)}w_{pj}^{(l+1)}z_h^{(l,k)}(1-z_h^{(l,k)})(-1)=-z_h^{(l,k)}(1-z_h^{(l,k)})\sum_{j=1}^{n_{l+1}}\delta_j^{(l+1,k)}w_{pj}^{(l+1)}
$$
对于阈值的梯度也可以这样想：
$$
\frac{\partial E}{\partial b_p^{(l)}}=\sum_{j=1}^{n_{l+1}}\frac{\partial E}{\partial \alpha_j^{(l+1,k)}}\cdot \frac{\partial \alpha_j^{(l+1,k)}}{\partial z_p^{(l,k)}}\frac{\partial z_p^{(l,k)}}{\partial b_p^{(l)}}
$$
**误差项计算**：

对于输出层第 $j$ 个神经元，第 $k$ 个样本：
$$
\delta_j^{(L,k)} = \frac{\partial E}{\partial \alpha_j^{(L,k)}} = \frac{1}{m} (\hat{y}_j^{(k)} - y_j^{(k)})\hat{y}_j^{(k)}(1-\hat{y}_j^{(k)})
$$
对于第$l(l=1,2\dots L-1)$层：
$$
\delta_p^{(l,k)} = \sum_{j=1}^{n_{l+1}} \delta_j^{(l+1,k)} \cdot \frac{\partial \alpha_j^{(l+1,k)}}{\partial z_p^{(l,k)}} \cdot \frac{\partial z_p^{(l,k)}}{\partial \alpha_p^{(l,k)}}=\sum_{j=1}^{n_{l+1}} \delta_j^{(l+1,k)}w_{pj}^{(l+1)}\cdot z_p^{(l,k)}(1-z_p^{(l,k)})=z_p^{(l,k)}(1-z_p^{(l,k)})\sum_{j=1}^{n_{l+1}} \delta_j^{(l+1,k)}w_{pj}^{(l+1)}
$$
可以看出， $\delta_j^{(L,k)}$就类似之前的两层前馈网络中的 $g$，$\delta_p^{(l,k)} $类似之前的$e$.

此外，可以发现 $\frac{\partial E}{\partial b_p^{(l)}}=-\delta_p^{(l,k)}$



也可以选择 $\varphi_p^{(l,k)}=\frac{\partial E}{\partial z_p^{(l,k)}}$作为迭代中的“误差项”。

**矩阵形式：**
$$
\boldsymbol{\Delta}^{(L)}=\frac 1m (\mathbf{\hat{Y}} -\mathbf Y)\odot \mathbf{\hat{Y}} \odot (1-\mathbf{\hat{Y}})\\
\boldsymbol{\Delta}^{(l)}=\mathbf{Z}^{(l)} \odot(1-\mathbf{Z}^{(l)})\odot(\mathbf{W}^{(l+1)}\mathbf{\Delta}^{(l+1)})
$$
梯度矩阵为：
$$
\frac{\partial E}{\partial \mathbf{W}^{(l)}}=\mathbf{Z}^{(l-1)}(\mathbf{\Delta}^{(l)})^T\\
\frac{\partial E}{\partial \mathbf{b}^{(l)}} =-\boldsymbol{\Delta}^{(l)} \mathbf{1}_m
$$
其中 $\mathbf{1}_m$ 是长度为 $m$ 的全1向量，结果维度为 $(n_l, 1)$。

**参数更新**:
$$
\mathbf{W}^{(l)}\leftarrow \mathbf{W}^{(l)}-\eta \mathbf{Z}^{(l-1)}(\mathbf{\Delta}^{(l)})^T\\
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} + \eta\sum_{\text{axis}=1} \boldsymbol{\Delta}^{(l)}
$$
从第$L$层开始迭代地更新参数即可。



## 手写数字识别

我让AI将上面的多层感知机用于手写识别，代码见 [test_mnist.py](test_mnist.py) 。结果如下：

```
正在加载 MNIST 数据集...
使用前 7000 个样本
训练集大小: 5600
测试集大小: 1400
输入维度: 784
训练数据形状: (784, 5600)
训练标签形状: (10, 5600)
输出类别数: 10

网络结构: [784, 64, 10]
正在创建神经网络...

开始训练...
Iteration 500, Loss: 0.027021507527322124
Iteration 1000, Loss: 0.011794440643460143
Iteration 1500, Loss: 0.007937120864037785
Iteration 2000, Loss: 0.006505801761123223
Iteration 2500, Loss: 0.0056441748474832295
Iteration 3000, Loss: 0.005033530192529117
达到最大迭代次数，最终误差为：0.005033530192529117
训练耗时: 90.42 秒

在训练集上评估...
训练集准确率: 94.93%

在测试集上评估...
测试集准确率: 91.71%
```



手搓的神经网络，并且是使用的最基础的sigmoid函数，网络结构也很简单，准确率可观，这表明我们成功构建了多层感知机。

采用更豪华的参数进行训练：

```
在加载 MNIST 数据集...
使用前 30000 个样本
训练集大小: 24000
测试集大小: 6000
输入维度: 784
训练数据形状: (784, 24000)
训练标签形状: (10, 24000)
输出类别数: 10

网络结构: [784, 128, 64, 10]
正在创建神经网络...

开始训练...
...
Iteration 21500, Loss: 0.0010050121356916066
Iteration 22000, Loss: 0.0009721867775974367
在第 22000 次迭代时收敛！
最终误差为：0.0009721867775974367
训练耗时: 5544.71 秒

在训练集上评估...
训练集准确率: 99.05%

在测试集上评估...
测试集准确率: 96.65%
```







