# RL 目录结构与类图

## 目录结构（按功能）

- `RL/grid_world.py`：核心环境类 `GridWorld`
- `RL/BellmanEquation/policy_evaluation.py`：策略评估脚本（函数式，无类）
- `RL/BellamOptimalEquation/value_iteration.py`：值迭代类 `ValueIteration`
- `RL/TD learning/TD-Linear_example.py`：TD 线性状态价值近似脚本（函数式，无类）
- `RL/TD learning/sarsa_with_linear_function_approximation.py`：SARSA 线性动作价值近似，含配置类 `SARSAConfig`

> 说明：当前 RL 代码主要是“一个核心环境类 + 多个算法脚本”。  
> 严格意义上的类图只包含实际定义的类，不把普通函数当作类节点。

## Mermaid 类图

```mermaid
classDiagram
direction LR

class GridWorld {
  +int rows
  +int cols
  +int n_states
  +float gamma
  +list actions
  +int n_actions
  +ndarray policy
  +tuple target
  +set forbidden
  +set_target(target_pos)
  +set_forbidden(forbidden_pos)
  +set_rewards(r_boundary, r_forbidden, r_target)
  +state_to_index(state) int
  +index_to_state(index) tuple
  +step(state, action, stay_on_forbidden) tuple
  +sample_next(state, rng) tuple
  +build_linear_system() tuple
  +true_value() ndarray
  +render(ax)
  +render_with_state_value(v, title, use_heatmap, cmap, ax)
  +render_with_policy(ax)
}

class ValueIteration {
  <<static>>
  +q_value(env, V, s, a) float
  +iterate(env, threshold, max_iter) tuple
  +extract_policy(env, best_actions) ndarray
}

class SARSAConfig {
  +int num_episodes
  +int max_steps_per_episode
  +float alpha
  +float gamma
  +float epsilon_start
  +float epsilon_end
  +float epsilon_decay
  +int seed
}

ValueIteration ..> GridWorld : uses
SARSAConfig ..> GridWorld : used by SARSA training pipeline
```
