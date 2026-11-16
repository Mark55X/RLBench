"""
Prints the task table of the benchmark (markdown), from the classes themselves (no simulator needed).

    python -m rlbench.dynamic_bench.tools.describe
"""

from rlbench.dynamic_bench import TASKS, get_task_class


def main():
    print('| # | task | RLBench task | obstacle | static keyframes it blocks | ways out |')
    print('|---|---|---|---|---|---|')
    for i, name in enumerate(TASKS, 1):
        info = get_task_class(name).dynbench_info()
        print(f"| {i} | `{name}` | `{info['base_task']}` | {info['obstacle']} | {info['keyframe_segment']} | "
              f"{'; '.join(info['remedies'])} |")
    print()
    for name in TASKS:
        info = get_task_class(name).dynbench_info()
        print(f'### `{name}`\n')
        for level, note in info['levels'].items():
            print(f'- **{level}**: {note}')
        print()


if __name__ == '__main__':
    main()
