import numpy as np 
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

housing= datasets.fetch_california_housing()

#数据预处理
x= housing.data
y= housing.target
ss= StandardScaler()
x= ss.fit_transform(x)

#划分训练集和测试集
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
df= pd.DataFrame(x_train,columns=housing.feature_names)
print(df.head())

#回归模型
#线性回归
def linear_regression():
    model= linear_model.LinearRegression()
    return model

model1= linear_regression()
model1.fit(x_train,y_train)
train_score1= model1.score(x_train,y_train)
test_score1= model1.score(x_test,y_test)
print('线性回归训练集得分：',train_score1)
print('线性回归测试集得分：',test_score1)
y_pred1= model1.predict(x_test)
rsme1= np.sqrt(mean_squared_error(y_test,y_pred1))
print('线性回归测试集均方根误差：',rsme1)

#lasso回归
def lasso_regression():
    model= linear_model.Lasso(alpha=0.1)
    return model

model2= lasso_regression()
model2.fit(x_train,y_train)
train_score2= model2.score(x_train,y_train)
test_score2= model2.score(x_test,y_test)
print('lasso回归训练集得分：',train_score2)
print('lasso回归测试集得分：',test_score2)
y_pred2= model2.predict(x_test)
rsme2= np.sqrt(mean_squared_error(y_test,y_pred2))
print('lasso回归测试集均方根误差：',rsme2)

#多项式回归
# 定义一个多项式回归函数，默认度为1
def polynomial_regression(degree=1):
    # 创建一个管道，包含多项式特征和线性回归模型
    model= Pipeline([
        ('poly_features',PolynomialFeatures(degree=degree)),  # 多项式特征
        ('poly_reg',linear_model.LinearRegression())  # 线性回归模型
    ])
    # 返回模型
    return model

model3= polynomial_regression(degree=2)
model3.fit(x_train,y_train)
train_score3= model3.score(x_train,y_train)
test_score3= model3.score(x_test,y_test)
print('多项式回归训练集得分：',train_score3)
print('多项式回归测试集得分：',test_score3)
y_pred3= model3.predict(x_test)
rsme3= np.sqrt(mean_squared_error(y_test,y_pred3))
print('多项式回归测试集均方根误差：',rsme3)

#结果可视化
plt.scatter(y_test,y_pred3,s=2)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='k')
plt.show()