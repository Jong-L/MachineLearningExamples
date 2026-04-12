[TOC]



# 两层前馈神经网络

## 网络结构

- 输入层：$d$ 个神经元
- 隐藏层：$q$ 个神经元，激活函数 $\sigma$
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
| $\hat{y}_j$ | 标量 | 输出层第 $j$ 个神经元的输出 |
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

单个样本的误差：
$$E_k = \frac{1}{2} \sum_{j=1}^{l} (\hat{y}_j-y_j)^2$$

#### 反向传播

输出层梯度项：
$$g_j = \hat{y}_j (1 - \hat{y}_j) (y_j - \hat{y}_j)$$

隐藏层梯度项：
$$e_h = b_h (1 - b_h) \sum_{j=1}^{l} w_{hj} g_j$$

#### 参数更新

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



## BGD

神经网络的最终目的是最小化$\frac 12 \mathbb{E}[(\hat{Y}-Y)^2]$,假设对于每个样本出现概率完全相同（比如函数拟合问题），则最小化$\frac {1}{2m} \sum_{k=1}^m (\hat{y^k}-y^k)^2$

按照这个目标函数，参数更新量为：

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

BGD的代码实现可以比较轻易地改为MBGD，以B=2为例，每次从$X_{d\times m}$中随机选取两列组成新矩阵作为输入矩阵，相应的也从$Y_{l\times m}$得到输出矩阵，用它们更新一次网络的参数，然后选择剩下的两列，然后循环。可以把BGD当成$B=m$的MBGD。

# 任意层数的前馈神经网络（BGD）



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

也可以选择 $\varphi_p^{(l,k)}=\frac{\partial E}{\partial z_p^{(l,k)}}$作为迭代中的“误差项”，以后再来讨论用这个的问题。

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









