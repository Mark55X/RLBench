"""
The STATIC keyframes of an episode: what a keyframe policy trained on the static task would decide a priori, on the
scene before the person changes it.

They are the keyposes of the static demo of the same episode (same seed: same layout), found with PerAct's keypoint
discovery (gripper state changes and stops of the arm), the action distribution PerAct / RVT / BridgeVLA are trained
on. The demo is recorded with the person and the benchmark's items out of the scene (core.DynamicBenchTask), so the
keyframes never account for the obstacles: the layer between keyframes has to.

    from rlbench.dynamic_bench.keyframes import static_episode
    descriptions, obs, demo, keyframes = static_episode(task_env, seed=3)
    # keyframes[k] = {'index', 'position', 'quaternion' (x, y, z, w; Panda_tip frame), 'gripper_open',
    #                 'ignore_collisions', 'action' (PerAct vector [x y z qx qy qz qw open ignore_collisions])}

static_episode also tells the task which keyframes the robot follows (dynbench_set_keyframes), for the metrics.
Without a learned policy, these keyframes let every benchmark task be run with any keyframe follower.
"""

from typing import List, Tuple

import numpy as np


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1) -> bool:
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
        i < (len(demo) - 2)
        and (obs.gripper_open == demo[i + 1].gripper_open and obs.gripper_open == demo[i - 1].gripper_open
             and demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return stopped_buffer <= 0 and small_delta and (not next_is_not_final) and gripper_state_no_change


def discover_keypoints(demo, stopping_delta: float = 0.1) -> List[int]:
    """PerAct's keypoint discovery: demo steps where the gripper opens / closes or the arm stops, and the last one."""
    keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(keypoints) > 1 and (keypoints[-1] - 1) == keypoints[-2]:
        keypoints.pop(-2)
    return keypoints


def keyframes_from_demo(demo) -> List[dict]:
    """Gripper pose and state at each keypoint, with the ignore_collisions flag of the step before it (PerAct)."""
    out = []
    for i in discover_keypoints(demo):
        obs, prev = demo[i], demo[max(0, i - 1)]
        pose = np.asarray(obs.gripper_pose, dtype=np.float64)
        q = pose[3:7] / np.linalg.norm(pose[3:7])
        opened = float(obs.gripper_open > 0.5)
        ignore = float(bool(getattr(prev, 'ignore_collisions', 0.0)))
        out.append({'index': int(i), 'position': [float(x) for x in pose[:3]], 'quaternion': [float(x) for x in q],
                    'gripper_open': opened, 'ignore_collisions': ignore,
                    'action': [float(x) for x in pose[:3]] + [float(x) for x in q] + [opened, ignore]})
    return out


def record_static_demo(task_env, attempts: int = 3):
    """A live demo of the static task (the benchmark task takes the person and its items out while RLBench records
    it)."""
    last = None
    for _ in range(attempts):
        try:
            return task_env.get_demos(1, live_demos=True, callable_each_step=lambda *a, **k: None)[0]
        except Exception as e:  # RLBench demos fail now and then
            last = e
    raise RuntimeError(f'demo failed: {last}')


def static_episode(task_env, seed: int) -> Tuple[list, object, object, List[dict]]:
    """Starts the episode of a seed and returns (descriptions, first observation, static demo, static keyframes): the
    demo is recorded on the static scene of that seed, the episode is then reset to it (same layout, and the person
    and the obstacles of the level), and the task is told the keyframes."""
    np.random.seed(seed)
    task_env.reset()
    np.random.seed(seed)
    demo = record_static_demo(task_env)
    descriptions, obs = task_env.reset_to_demo(demo)
    keyframes = keyframes_from_demo(demo)
    task = task_env._task
    if hasattr(task, 'dynbench_set_keyframes'):
        task.dynbench_set_keyframes(keyframes)
    return descriptions, obs, demo, keyframes
