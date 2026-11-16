"""
Dynamic benchmark on RLBench: obstacles that come up AFTER a keyframe policy has decided. Tasks on which BridgeVLA++
(and PerAct / RVT) are trained, unchanged; their static keyframes (keyframes.py) stay valid, and a person seated at the
table puts things in the robot's way between keyframes, on keyframe targets, or carries them across the robot's way.
See README.md.

    from rlbench.dynamic_bench import TASKS, get_task_class, set_level
    from rlbench.dynamic_bench.keyframes import static_episode
    set_level('L1')                                   # or the environment variable DYNBENCH_LEVEL
    task_env = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
    descriptions, obs, demo, keyframes = static_episode(task_env, seed=0)
"""

import importlib
from typing import Dict, List

from rlbench.dynamic_bench.core import (LEVELS, DynamicBenchTask, current_human_speed_scale, current_level,
                                        normalize_level, set_human_speed_scale, set_level)

# Benchmark task name -> class name, in the order of the README
TASKS: Dict[str, str] = {
    'dyn_stack_blocks_goal_taken': 'DynStackBlocksGoalTaken',
    'dyn_stack_blocks_tunnel': 'DynStackBlocksTunnel',
    'dyn_close_jar_item_in_path': 'DynCloseJarItemInPath',
    'dyn_shape_sorter_book_on_hole': 'DynShapeSorterBookOnHole',
    'dyn_place_wine_bottle_in_path': 'DynPlaceWineBottleInPath',
    'dyn_put_groceries_grasp_blocked': 'DynPutGroceriesGraspBlocked',
    'dyn_slide_block_obstacle_in_path': 'DynSlideBlockObstacleInPath',
    'dyn_slide_block_target_taken': 'DynSlideBlockTargetTaken',
    'dyn_slide_block_book_across': 'DynSlideBlockBookAcross',
    'dyn_insert_peg_hand_in_way': 'DynInsertPegHandInWay',
}


# tasks defined in another task's module
MODULES: Dict[str, str] = {'dyn_slide_block_target_taken': 'dyn_slide_block_obstacle_in_path'}


def task_names() -> List[str]:
    return list(TASKS)


def get_task_class(name: str):
    """The task class of a benchmark task name (file name in rlbench/dynamic_bench/tasks)."""
    name = name.replace('.py', '')
    if name not in TASKS:
        raise KeyError(f"'{name}' is not a dynamic benchmark task (known: {', '.join(TASKS)})")
    module = importlib.import_module(f'rlbench.dynamic_bench.tasks.{MODULES.get(name, name)}')
    return getattr(module, TASKS[name])


def is_benchmark_task(name: str) -> bool:
    return name.replace('.py', '') in TASKS
