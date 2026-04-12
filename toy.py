import numpy as np

x = 3
a = np.array([x > 0]).astype(float)[0]  # 创建NumPy数组然后转换
print(a)