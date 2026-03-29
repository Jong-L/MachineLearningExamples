增量学习估计状态值，TD（0）
TD(0)是为了估计状态值，即$E[v_\pi(S)]=E[R+\gamma v_\pi(S')]$
设$f(w)=w-E[R+\gamma v_\pi(S')]$
即求解：$f(w)=0$
采用RM算法，设定$\eta=E[R+\gamma v_\pi(S')]-v_\pi(s')=E[R+\gamma v_\pi(S')]-（r+\gamma v_\pi(s'))$
于是$\tilde{f(w,\eta)}=f(w)+\eta=w_k-（r+\gamma v_\pi(s'))$
$w$就是对$v_\pi(S)$的估计，将$w_{k}$作为$v_{k}(S)$
于是计算方法为：$v_{k+1}(s)=v_k(s)-\alpha_k(v_k(s)-(r+\gamma v_\pi(s')))$
在实际应用中由于不知道模型，将$v_\pi(s')$用$v_k(s')$代替,学习率采用一个小值$\alpha_k =\alpha$
