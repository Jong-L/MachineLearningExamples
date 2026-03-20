import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#prepare dataset
df=pd.read_excel('machine learning/dataset/watermelon3.0 alpha.xlsx')
print(df.head())
#均值向量
good_melon_0 = df[df['good melon'] == 0].iloc[:, :2].to_numpy()
u0 =np.mean(good_melon_0,axis=0)#沿列求均值
good_melon_1 = df[df['good melon'] == 1].iloc[:, :2].to_numpy()
u1 = np.mean(good_melon_1,axis=0)
u0=u0.reshape(-1,1)
u1=u1.reshape(-1,1)
a=good_melon_0.T-u0
b=good_melon_1.T-u1

#类内散度矩阵
sigma0=a@a.T
sigma1=b@b.T
Sw=np.add(sigma0,sigma1)
#奇异值分解
U, S, V = np.linalg.svd(Sw)

Sw_inv=V@np.linalg.inv(np.diag(S))@U.T


w=Sw_inv@(u0-u1)
print(w)
x=np.arange(0,0.5,step=0.05)
y=-w[0]/w[1]*x

sns.set_style('whitegrid')
sns.lmplot(x='density',y='sugar content',data=df,hue='good melon',fit_reg=False, scatter_kws={"s": 20})
plt.plot(x,y,linewidth=1.5)
plt.show()
