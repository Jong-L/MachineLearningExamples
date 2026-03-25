"""
由于整个示例中采用的环境比较简单，直接根据策略求解状态值对算力要求也不高，所以也给出策略迭代的示例
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid_world import GridWorld


