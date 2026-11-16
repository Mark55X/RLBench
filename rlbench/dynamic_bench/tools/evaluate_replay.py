"""
Reference baseline of the benchmark, needing no learned model: the joint trajectory of a live demo of the STATIC
task is replayed open loop in the dynamic scene with the same layout ('plan once on the static world, execute
blind'). With --level off it measures how often the replay itself works (the ceiling of the baseline).

One JSON line per episode goes to --out (append), and a summary table is printed at the end.

    python -m rlbench.dynamic_bench.tools.evaluate_replay --tasks all --levels off L0 L1 L2 --episodes 10 \
        --out results/replay.jsonl
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from rlbench.dynamic_bench import TASKS, get_task_class, set_level
from rlbench.dynamic_bench.tools.common import make_env, record_static_demo, replay_open_loop


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['task'], r['level'])].append(r)
    lines = ['| task | level | n | success | safe success | person contact rate | side effects | '
             'mean min clearance to the person |',
             '|---|---|---|---|---|---|---|---|']
    for (task, level), rs in sorted(groups.items()):
        ok = [r for r in rs if 'error' not in r]
        if not ok:
            lines.append(f'| {task} | {level} | 0 | error | | | | |')
            continue
        succ = np.mean([r['success'] for r in ok])
        safe = np.mean([r['success'] and not r['dynbench'].get('unsafe', r['dynbench']['collided']) for r in ok])
        coll = np.mean([r['dynbench']['collided'] for r in ok])
        held = np.mean([bool(r['dynbench']['toppled_by_robot'] or r['dynbench']['fell_off_by_robot']) for r in ok])
        clear = [r['dynbench']['min_clearance_person'] for r in ok]
        lines.append(f'| {task} | {level} | {len(ok)} | {succ:.2f} | {safe:.2f} | {coll:.2f} | {held:.2f} | '
                     f'{(np.mean(clear) if clear else float("nan")):.3f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', nargs='+', default=['all'])
    ap.add_argument('--levels', nargs='+', default=['off', 'L0', 'L1', 'L2'])
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0, help='episode e of a task uses numpy seed (seed + e)')
    ap.add_argument('--variation', type=int, default=-1, help='-1: random variation per episode')
    ap.add_argument('--out', default='dynbench_replay.jsonl')
    args = ap.parse_args()
    tasks = list(TASKS) if args.tasks == ['all'] else args.tasks

    env = make_env()
    rows = []
    for name in tasks:
        task_env = env.get_task(get_task_class(name))
        for level in args.levels:
            set_level(level)
            for e in range(args.episodes):
                np.random.seed(args.seed + e)
                row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'task': name, 'level': level, 'episode': e,
                       'seed': args.seed + e, 'baseline': 'static_demo_open_loop_replay'}
                try:
                    if args.variation >= 0:
                        task_env.set_variation(args.variation)
                    else:
                        task_env.sample_variation()
                    row['variation'] = task_env._variation_number
                    demo = record_static_demo(task_env)
                    descriptions, _ = task_env.reset_to_demo(demo)
                    row['description'] = descriptions[0]
                    ok, steps, _ = replay_open_loop(task_env, demo)
                    row.update(success=ok, steps=steps, demo_steps=len(demo),
                               dynbench=task_env._task.dynbench_metrics())
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
                rows.append(row)
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(row, default=str) + '\n')
                print(f"{name} {level} ep{e}: " + (row['error'] if 'error' in row else
                      f"success={row['success']} person_contacts={row['dynbench']['person_contact_events']} "
                      f"side_effects={row['dynbench']['toppled_by_robot'] + row['dynbench']['fell_off_by_robot']}"),
                      flush=True)
    set_level(None)
    print()
    print(summarize(rows))
    env.shutdown()


if __name__ == '__main__':
    main()
