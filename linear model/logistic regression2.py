import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def feature_mapping(x,y,power,as_ndarray=False):
    data={"f{}{}".format(i-p,p):np.power(x,i-p)*np.power(y,p)
          for i in np.arange(power+1)
          for p in np.arange(i+1)
    }
    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)
#数据部分
df=pd.read_csv('machine learning/dataset/ex2data2.txt',delimiter=',',names=['test1','test2','accept'])
sns.set(context='notebook',style='ticks',font_scale=1.5)
sns.lmplot(x='test1',y='test2',hue='accept',data=df,fit_reg=False,scatter_kws={'s':15}, palette={0: '#00FF00', 1: '#00BFFF'})
plt.show()
x1=np.array(df.test1)
x2=np.array(df.test2)
data=feature_mapping(x1,x2,power=6)
print(data.shape)
