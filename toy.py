import numpy as np
a = np.zeros(3)
b = np.zeros(3)
a[0] = 99
b=np.copy(a)
a[1]=3
print(a)  # 输出 [99  2  3]，说明 a 也被修改了
print(b)  # 输出 [99  2  3]，说明 b 也被修改了
