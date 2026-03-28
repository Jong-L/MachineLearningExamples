"""
Truncated Policy Iteration for Grid World
将policy_iteration.py中的new_v=env.get_true_value_by_policy(policy)
改为new_v=env.get_itrated_value_by_policy(policy,max_iterations: int =truncated_iterations)即可
这里不写了。
"""