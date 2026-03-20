#Iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import classification_report
import seaborn as sns
import scipy.optimize as opt

#sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(beta,X,Y):
    YHat=sigmoid(X.dot(beta))
    YHat=YHat.reshape(-1,1)
    return X.T@(YHat-Y)

def second_derivative(X,beta):
    YHat=sigmoid(X.dot(beta))
    YHat=YHat.reshape(-1,1)
    return X.T@(X*YHat*(1-YHat))

def costFunction(beta,X,Y):
    return np.sum(-Y*X@beta-np.log(sigmoid(-X@beta)))

def predict(X,beta):
    prob=sigmoid(X@beta)
    return (prob>=0.5).astype(int)

data=np.loadtxt("machine learning/dataset/ex2data1.txt",delimiter=",")
data = pd.DataFrame(data, columns=['density', 'sugar content', 'good melon?'])
data['good melon?'] = data['good melon?'].astype(int)

X=data.iloc[:,0:2].to_numpy()
Y=data.iloc[:,2].to_numpy()
Y=Y.reshape(-1,1)

X_mean=np.mean(X,axis=0)
X_std=np.std(X,axis=0)#方差
X=(X-X_mean)/X_std#标准化


X=np.insert(X,2,values=1,axis=1)
print(X)

beta=np.ones(3)
beta=beta.reshape(-1,1)
print(costFunction(beta,X,Y))

learningRate=0.1
for i in range(10000):
    beta=beta-np.linalg.inv(second_derivative(X,beta))@gradient(beta,X,Y)

print('new beta:',beta)
print('new cost:',costFunction(beta,X,Y))
coef=[-X_std[1]*beta[0]/X_std[0]/beta[1],X_mean[1]+X_std[1]/beta[1]*(-beta[2]+beta[0]*X_mean[0]/X_std[0])]
#coef=-beta/beta[1]
xp=np.arange(0,100,5)
yp=xp*coef[0]+coef[1]
#yp=100*coef[2]+coef[0]*xp
'''
采用了每个特征值都除100的方法，
评估结果与标准化一致，
显然，逻辑回归并不是特征放缩的用武之地
'''
sns.set_style('whitegrid')
sns.lmplot(x='density',y='sugar content',data=data,hue='good melon?',fit_reg=False, scatter_kws={"s": 20})
plt.plot(xp,yp, linewidth=1.5)
plt.show()
y_pred=predict(X,beta)
print(classification_report(Y,y_pred))