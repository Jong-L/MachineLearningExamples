策略为环境本身的随机策略
用参数$\theta$的线性函数近似状态值
特征向量为$\phi(s)=\phi(x,y)=[1, x, y, x \cdot y, x^2, y^2]^T$，其中$x$和$y$分别表示状态$s$的行和列坐标。
$\hat{v_\pi(s,\theta)}=\phi(s)^T\cdot \theta$
需要求解最优化问题：
$$
\min J(\theta)=\frac 12 E[(\hat{v_\pi(S,\theta)}-v_\pi(S))^2]
$$
这是一个凸优化问题，只需要求解梯度为0：
$$
\theta^* \quad s.t.\quad  E[\hat{v_\pi(S,\theta)}-v_\pi(S)]\nabla \hat{v_\pi(S,\theta)}=0
$$
进一步，即求解：
$$
g(\theta)=E[\hat{v_\pi(S,\theta)}-v_\pi(S)]\phi(S)=0
$$
采用随机梯度下降求解：
$$
\theta_{t+1}=\theta_t-\alpha_t \phi(s)(\hat{v_\pi(s,\theta)}-v_\pi(s))
$$
在这一节中，$v_\pi(s)$用$r_{t+1}+\gamma \hat{v_\pi}(s_{t+1},\theta)$代替。
