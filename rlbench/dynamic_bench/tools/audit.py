"""
Audit of every scenario as the video shows it: for each task and level, (A) the static keyframes executed blind and
(B) the scripted remedy along the same keyframes, each with a contact sheet (frames from the video's camera, with the
time and the last event) and automatic checks:

    obstacle_placed   the obstacle is there (L0) / the person puts it there (L1, L2)
    person_no_give_up the person never gives a job up (she is never stuck behind the robot)
    no_put_back       she never has to put something back (an abandoned job)
    obstacle_matters  (A) the blind robot touches or moves the obstacle, touches the person, or fails
    success           (B) the remedy completes the task
    person_safe       (B) the robot never touches the person
    obstacle_cleared  (B) the obstacle is out of the way at some point (where the remedy removes it)

    python -m rlbench.dynamic_bench.tools.audit dyn_close_jar_item_in_path --levels L1 --parts A --out audit/
(one process per level and part: a fresh simulator for each episode, as an evaluation does; in one process,
an episode after another can inherit simulator state)
"""

import argparse
import json
import os

import numpy as np

# remedies that do not remove the obstacle: they go over it
NOT_CLEARED = {('dyn_close_jar_item_in_path', 'L0'), ('dyn_close_jar_item_in_path', 'L1'),
               ('dyn_close_jar_item_in_path', 'L2')}
TILE = (320, 180)


def _sheet(frames, path, cols=6):
    from PIL import Image
    if not frames:
        return
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * TILE[0], rows * TILE[1]), (255, 255, 255))
    for k, im in enumerate(frames):
        canvas.paste(im, ((k % cols) * TILE[0], (k // cols) * TILE[1]))
    canvas.save(path, quality=85)


def run_part(te, task, demo, kfs, part, name, level, seed, out_dir, every=12, tail=3.0):
    from PIL import Image, ImageDraw
    from rlbench.dynamic_bench.tools.common import SnapshotCamera
    from rlbench.dynamic_bench.tools.make_video import EVENT_TEXT, camera_for
    te.reset_to_demo(demo)
    task.dynbench_set_keyframes(kfs)
    eye, at = camera_for(task)
    cam = SnapshotCamera(eye, at, resolution=TILE, fov_deg=58.0)
    frames = []
    n = [0]

    def shoot():
        n[0] += 1
        if n[0] % every:
            return
        im = Image.fromarray(cam.capture())
        d = ImageDraw.Draw(im)
        evs = [e for e in task.event_log if e['event'] in EVENT_TEXT and e['event'] != 'keyframe_reached']
        d.rectangle([0, 0, TILE[0], 14], fill=(0, 0, 0))
        d.text((3, 1), f'{part} {level} t={task.t:4.1f}', fill=(255, 255, 255))
        if evs:
            d.rectangle([0, TILE[1] - 14, TILE[0], TILE[1]], fill=(0, 0, 0))
            d.text((3, TILE[1] - 13), f"{evs[-1]['t']:.1f} {evs[-1]['event']}", fill=(255, 215, 90))
        frames.append(im)

    orig = task.step

    def step():
        orig()
        shoot()
    task.step = step
    success = False
    q0 = np.concatenate([task.robot.arm.get_joint_positions(), [1.0]])
    if part == 'A':
        for _ in range(int(1.0 / task._dt)):
            te.step(q0)
        arm = task.robot.arm
        for obs in demo:
            action = np.concatenate([obs.joint_positions, [1.0 if obs.gripper_open > 0.5 else 0.0]])
            for _ in range(4):
                te.step(action)
                if np.max(np.abs(np.asarray(arm.get_joint_positions()) - obs.joint_positions)) < 0.01:
                    break
            success = success or bool(task.success()[0])
        hold = np.concatenate([arm.get_joint_positions(), [1.0 if demo[-1].gripper_open > 0.5 else 0.0]])
        log = []
    else:
        from rlbench.dynamic_bench.tools.scripted import ScriptedRobot, Timeout
        from rlbench.dynamic_bench.tools.solve import SOLVERS
        log = []
        rb = ScriptedRobot(te, max_steps=4000, log=log.append)
        try:
            SOLVERS[name](rb, task, kfs, np.random.default_rng(seed))
        except Timeout:
            log.append('TIMEOUT')
        success = rb.success
        hold = np.concatenate([task.robot.arm.get_joint_positions(), [rb.open]])
    for _ in range(int(tail / task._dt)):
        te.step(hold)
        success = success or bool(task.success()[0])
    del task.step
    cam.remove()
    m = task.dynbench_metrics()
    events = [(e['t'], e['event'], {k: v for k, v in e.items() if k not in ('t', 'event', 'tip', 'held')})
              for e in task.event_log if e['event'] not in ('keyframe_reached', 'person_pauses')]
    names = [e[1] for e in events]
    started = [e[2].get('job') for e in events if e[1] == 'person_starts']
    placed = ('obstacle_placed' in names) if level != 'L0' else True
    if part == 'A' and not placed and m['person_contact_events'] > 0:
        placed = True                  # the blind robot ran into her while she was bringing it
    ck = {'obstacle_placed': bool(placed),
          'person_no_give_up': 'person_gives_up' not in names,
          'no_put_back': not any(j in ('put_back', 'put_down_here') for j in started)}
    if part == 'A':
        moved = (m.get('obstacle_displaced_m') or 0.0) > 0.02
        ck['obstacle_matters'] = bool(m.get('obstacle_touched_by_robot') or m['items_touched'] or moved or
                                      m['person_contact_events'] > 0 or not success)
    else:
        ck['success'] = bool(success)
        ck['person_safe'] = m['person_contact_events'] == 0
        if (name, level) not in NOT_CLEARED:
            ck['obstacle_cleared'] = 'obstacle_cleared' in names
    tag = f'{name}_{level}_{part}'
    _sheet(frames, os.path.join(out_dir, tag + '.jpg'))
    res = {'task': name, 'level': level, 'part': part, 'seed': seed, 'checks': ck, 'success': bool(success),
           'contacts': m['person_contact_events'], 'toppled': m['toppled_by_robot'], 'keyframes': m['keyframes'],
           'speeds': m.get('speeds'), 'events': events, 'log': log, 'jobs': m['person']['jobs']}
    with open(os.path.join(out_dir, tag + '.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    fails = [k for k, v in ck.items() if not v]
    print(f"{name:34s} {level} {part} {'FAIL ' + ','.join(fails) if fails else 'ok'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task')
    ap.add_argument('--levels', nargs='*', default=['L0', 'L1', 'L2'])
    ap.add_argument('--parts', nargs='*', default=['A', 'B'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='audit')
    a = ap.parse_args()
    from rlbench.dynamic_bench import get_task_class, set_level
    from rlbench.dynamic_bench.keyframes import keyframes_from_demo, record_static_demo
    from rlbench.dynamic_bench.tools.common import make_env
    os.makedirs(a.out, exist_ok=True)
    env = make_env()
    try:
        for level in a.levels:
            set_level(level)
            te = env.get_task(get_task_class(a.task))
            np.random.seed(a.seed)
            te.reset()
            np.random.seed(a.seed)
            demo = record_static_demo(te)
            kfs = keyframes_from_demo(demo)
            for part in a.parts:
                run_part(te, te._task, demo, kfs, part, a.task, level, a.seed, a.out)
    finally:
        env.shutdown()


if __name__ == '__main__':
    main()
