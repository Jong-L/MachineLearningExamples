import sympy as sp

# 定义符号
x, y = sp.symbols('x y')

# 定义表达式
expression = x**2 + y**2

# 创建一个字典，将符号替换为数值
values = {x: 3, y: 4}

# 使用subs方法替换符号并计算表达式的值
result = expression.subs(values)

# 输出结果
print(result)  # 输出结果应该是 3^2 + 4^2 = 9 + 16 = 25
