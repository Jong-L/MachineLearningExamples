
"""
最优解管理器模块
用于保存和加载通过值迭代或策略迭代得到的贝尔曼最优方程的解
想要比较model free的算法得到的策略如何，又不想每次都跑一遍值迭代或策略迭代，所以把结果保存下来。
"""
import os
import hashlib
import json
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from grid_world import GridWorld

# 解决结果保存目录
SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "optimal_solutions")

@dataclass
class OptimalSolution:
    """最优解数据结构"""
    env_id: str
    env_config: Dict
    value: np.ndarray
    policy: np.ndarray
    algorithm: str  # 'value_iteration' or 'policy_iteration'
    timestamp: str#什么时候保存的
    iterations: int#得到这个解的迭代次数
    delta: float#最后一次迭代时，值函数的差值
    converged: bool#是否收敛（误差是否小于设定的阈值）


def generate_env_id(env: GridWorld) -> str:
    """
    生成环境的唯一标识符

    参数:
        rows: 网格行数
        cols: 网格列数
        gamma: 折扣因子
        target: 目标位置
        forbidden: 禁止区域集合
        r_boundary: 边界奖励
        r_forbidden: 禁止区域奖励
        r_target: 目标奖励

    返回:
        环境ID字符串
    """
    # 将禁止区域排序并转换为元组列表，确保顺序一致
    forbidden_sorted = tuple[Tuple[int, int], ...](sorted(env.forbidden))

    # 创建配置字典
    config={
        'rows': env.rows,
        'cols': env.cols,
        'gamma': env.gamma,
        'target': env.target,
        'forbidden': forbidden_sorted,
        'r_boundary': env.r_boundary,
        'r_forbidden': env.r_forbidden,
        'r_target': env.r_target

    }

    # 将配置转换为JSON字符串
    config_str = json.dumps(config, sort_keys=True)

    # 生成MD5哈希作为环境ID
    env_id = hashlib.md5(config_str.encode()).hexdigest()

    return env_id


def get_solution_path(env_id: str) -> str:
    """
    获取解文件的保存路径

    参数:
        env_id: 环境ID

    返回:
        解文件的完整路径
    """
    if not os.path.exists(SOLUTIONS_DIR):
        os.makedirs(SOLUTIONS_DIR)

    return os.path.join(SOLUTIONS_DIR, f"solution_{env_id}.npz")


def save_optimal_solution(solution: OptimalSolution) -> str:
    """
    保存最优解到文件

    参数:
        solution: OptimalSolution对象

    返回:
        保存的文件路径
    """
    file_path = get_solution_path(solution.env_id)

    # 保存numpy数组和元数据
    np.savez_compressed(
        file_path,
        value=solution.value,
        policy=solution.policy,
        env_id=solution.env_id,
        env_config=json.dumps(solution.env_config),
        algorithm=solution.algorithm,
        timestamp=solution.timestamp,
        iterations=solution.iterations,
        delta=solution.delta,
        converged=solution.converged
    )

    return file_path


def load_optimal_solution(env_id: str) -> Optional[OptimalSolution]:
    """
    从文件加载最优解

    参数:
        env_id: 环境ID

    返回:
        OptimalSolution对象，如果文件不存在则返回None
    """
    file_path = get_solution_path(env_id)

    if not os.path.exists(file_path):
        return None

    # 加载数据
    data = np.load(file_path, allow_pickle=True)

    solution = OptimalSolution(
        env_id=str(data['env_id']),
        env_config=json.loads(str(data['env_config'])),
        value=data['value'],
        policy=data['policy'],
        algorithm=str(data['algorithm']),
        timestamp=str(data['timestamp']),
        iterations=int(data['iterations']),
        delta=float(data['delta']),
        converged=bool(data['converged'])
    )

    return solution


def has_optimal_solution(env_id: str) -> bool:
    """
    检查是否存在已保存的最优解

    参数:
        env_id: 环境ID

    返回:
        如果存在则返回True，否则返回False
    """
    file_path = get_solution_path(env_id)
    return os.path.exists(file_path)


def list_all_solutions() -> list:
    """
    列出所有已保存的解

    返回:
        解文件路径列表
    """
    if not os.path.exists(SOLUTIONS_DIR):
        return []

    solutions = []
    for file_name in os.listdir(SOLUTIONS_DIR):
        if file_name.startswith('solution_') and file_name.endswith('.npz'):
            solutions.append(os.path.join(SOLUTIONS_DIR, file_name))

    return solutions


def delete_solution(env_id: str) -> bool:
    """
    删除指定环境的最优解
    参数:
        env_id: 环境ID
    返回:
        如果删除成功返回True，否则返回False
    """
    file_path = get_solution_path(env_id)

    if not os.path.exists(file_path):
        return False

    try:
        os.remove(file_path)
        return True
    except Exception:
        return False
