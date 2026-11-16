"""
Pictures of a benchmark task over time, with the robot still (what the person does without the robot's progress:
the obstacle brought right after the start) and optionally during the open-loop replay of the static demo.

    python -m rlbench.dynamic_bench.tools.inspect_task dyn_close_jar_item_in_path --level L1 --out /tmp/shots
"""

import argparse
import json
import os

import numpy as np

from rlbench.dynamic_bench import get_task_class, set_level
from rlbench.dynamic_bench.tools.common import (SnapshotCamera, make_env, montage, record_static_demo,
                                                replay_open_loop, save_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task')
    ap.add_argument('--level', default='L1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variation', type=int, default=0)
    ap.add_argument('--idle_seconds', type=float, default=12.0, help='robot still for this long')
    ap.add_argument('--shots', type=int, default=6)
    ap.add_argument('--replay', action='store_true', help='then replay the static demo open loop')
    ap.add_argument('--eye', type=float, nargs=3, default=[1.35, -0.85, 1.55])
    ap.add_argument('--look_at', type=float, nargs=3, default=[0.25, 0.0, 0.85])
    ap.add_argument('--out', default='dynbench_shots')
    args = ap.parse_args()

    np.random.seed(args.seed)
    set_level(args.level)
    env = make_env()
    task_env = env.get_task(get_task_class(args.task))
    task_env.set_variation(args.variation)
    cam = SnapshotCamera(args.eye, args.look_at)
    os.makedirs(args.out, exist_ok=True)
    task = task_env._task
    descriptions, _ = task_env.reset()
    print('description:', descriptions[0])
    print('items:', {n: (None if o.hidden else np.round(o.position(), 3).tolist()) for n, o in task.items.items()
                     if not o.hidden})
    print('person:', {'side': task.person.side, 'seat_x': round(task.person.seat_x, 3)})

    images, labels = [cam.capture()], ['t=0 (start)']
    n = int(args.idle_seconds / task._dt)
    every = max(1, n // max(1, args.shots - 1))
    q = np.concatenate([task.robot.arm.get_joint_positions(), [1.0]])
    for k in range(1, n + 1):
        task_env.step(q)
        if k % every == 0:
            images.append(cam.capture())
            labels.append(f't={k * task._dt:.1f}s idle')
    save_png(montage(images, labels), os.path.join(args.out, f'{args.task}_{args.level}_idle.png'))
    print('idle metrics:', json.dumps({k: v for k, v in task.dynbench_metrics().items() if k != 'contact_log'}))

    if args.replay:
        demo = record_static_demo(task_env)
        task_env.reset_to_demo(demo)
        print('items after reset_to_demo:',
              {nm: np.round(o.position(), 3).tolist() for nm, o in task.items.items() if not o.hidden})
        ok, steps, shots = replay_open_loop(task_env, demo, snapshot=cam, snapshot_every=max(1, len(demo) // 8))
        save_png(montage(shots, [f'replay step {(i + 1) * max(1, len(demo) // 8)}' for i in range(len(shots))]),
                 os.path.join(args.out, f'{args.task}_{args.level}_replay.png'))
        m = task.dynbench_metrics()
        print(f'replay: success={ok} steps={steps} demo_len={len(demo)}')
        print('replay metrics:', json.dumps(m, default=str))
    env.shutdown()


if __name__ == '__main__':
    main()
