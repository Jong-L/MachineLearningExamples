import pandas as pd
import numpy as np
# 加载数据集
data = pd.read_csv('machine learning/dataset/car+evaluation/car.data', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']#价格，维护费用，车门数量，乘客数量，行李箱空间，安全性

class DecisionNode:
    '''
    若不为叶子结点
    attribute: 当前判断属性
    values: 属性所有可选值[value1,value2...]
    若为叶子结点
    attribute: 标签名
    values: 该分支对应标签值value
    '''
    def __init__(self, attribute=None, values=None, isleaf=False):
        self.a = attribute
        self.values = values
        self.isleaf = isleaf
