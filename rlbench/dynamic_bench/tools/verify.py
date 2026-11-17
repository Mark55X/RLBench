"""
Verification of the benchmark SIMULATION (does every scene behave as designed?), not of a robot policy:
    python -m rlbench.dynamic_bench.tools.verify [--tasks a b] [--seeds 0 1 2] [--parts static blind solve]
                                                 [--out results/verify.json]

static (every level, every seed; the robot stands still):
    placed          the episode can be placed (layout rejections are counted)
    same_layout     L1/L2 without rejections: the task objects are exactly where the static task puts them (level off,
                    same seed), so results compare with the static task episode by episode
    still_inactive  before the client activates the episode nothing of the benchmark moves
    obstacle        L0 / L1: the obstacle is there (L0 from the start, L1 brought by the person) and in the way of the
                    static keyframes; L2: the person carries things across the robot's way (traffic) where the task has
                    traffic (the L2 obstacle itself waits for the robot's progress)
    supported       at the end, every free item rests on the table or on another object, at rest: nothing floats
    no_explosion    no free body ever moves faster than 2.5 m/s, none ends below the table top
    oracle          every benchmark body has a dynamics class: the person 'agent', the items 'passive' / 'carried'
    idle_safe       L1/L2: the person never touches a robot that stands still (20 s)
blind (the static keyframes executed blind: the static demo replayed open loop, L1 and L2, seed 0): the episode runs,
    no explosion, ground truth recorded, and the obstacle MATTERS: the blind robot touches it or the person, or fails
solve (scripted remedy that follows the same static keyframes, tools/solve.py, L1 and L2, seed 0): the scenario's
    events happen in the designed order, the task is completed, the person is never touched
"""

import argparse
import json
import os
import time
import traceback
from typing import Dict, List

import numpy as np

from rlbench.dynamic_bench import TASKS, get_task_class, set_level
from rlbench.dynamic_bench.core import TABLE_Z

MAX_SPEED = 2.5            # [m/s] faster than this is a physics explosion, not something a person or a robot does
# objects that move by themselves in the ORIGINAL static task, measured (python: idle robot, 20 s, seeds 0-7): the
# cylinder of place_shape_in_shape_sorter rolls away in 4 of 8 seeds (off the table in one), RLBench spawns it on its
# side, and hits the other shapes; their behaviour is the static task's, so they are left out of the checks (the
# benchmark's own items in that task are checked)
KNOWN_UNSTABLE = {'place_shape_in_shape_sorter': {'cylinder', 'triangular_prism', 'star', 'moon', 'cube'}}
SUPPORT_GAP = 0.006        # [m] an item rests on something when it is this close to it (or to the table top)
REST_SPEED = 0.03          # [m/s]

# events that must happen, in this order, in the scripted remedy of each task (both levels), and at L2 also these
SCENARIOS: Dict[str, List[str]] = {
    'dyn_stack_blocks_goal_taken': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_close_jar_item_in_path': ['obstacle_coming', 'obstacle_placed'],
    'dyn_shape_sorter_tablet_on_hole': ['obstacle_coming', 'obstacle_placed', 'shape_parked', 'obstacle_cleared'],
    'dyn_place_wine_bottle_in_path': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_put_groceries_grasp_blocked': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_slide_block_obstacle_in_path': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_insert_peg_hand_in_way': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_stack_blocks_tunnel': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_shape_sorter_box_over_shape': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
    'dyn_slide_block_board_pinned': ['obstacle_coming', 'obstacle_placed', 'obstacle_cleared'],
}
SCENARIOS_L2: Dict[str, List[str]] = {
    'dyn_stack_blocks_goal_taken': ['traffic_crossing', 'block_parked', 'second_blocker_coming'],
    'dyn_close_jar_item_in_path': ['traffic_crossing'],
    'dyn_place_wine_bottle_in_path': ['traffic_crossing'],
}
TRAFFIC = set(SCENARIOS_L2)
HAND_TASKS = {'dyn_insert_peg_hand_in_way'}


def _bodies(task):
    """Free bodies to watch: the items and the task's objects under physics now (not what the person carries)."""
    from pyrep.const import ObjectType
    out = {it.name: it.shape for it in task.items.values() if not it.hidden and it.state == 'passive'}
    for s in task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE):
        if s.is_respondable() and s.is_dynamic():
            out[s.get_name()] = s
    return out


def _supported(task, it) -> bool:
    from rlbench.dynamic_bench.assets import _mesh_world
    low = float(_mesh_world(it.shape)[:, 2].min())
    if -0.004 <= low - TABLE_Z <= SUPPORT_GAP:
        return True
    return task.near(it.shape, SUPPORT_GAP, ignore=set(it.handles())) is not None or \
        task.overlaps(it.shape, robot=False, ignore=set(it.handles())) is not None


class Watch:
    """Max speed and lowest height of every watched body over the steps."""

    def __init__(self, task, ignore=()):
        self.task = task
        self.ignore = set(ignore)
        self.prev = {}
        self.max_speed = {}
        self.min_z = {}
        self.t = None

    def step(self, _=None) -> None:
        t = self.task.t
        dt = None if self.t is None else t - self.t
        self.t = t
        if dt is not None and dt <= 0:
            return
        bodies = {n: s for n, s in _bodies(self.task).items() if n not in self.ignore}
        for name in list(self.prev):
            if name not in bodies:
                del self.prev[name]            # carried / hidden for a while: no speed across the gap
        for name, s in bodies.items():
            p = np.asarray(s.get_position())
            if name in self.prev and dt is not None and np.linalg.norm(p - self.prev[name]) < 1.0:
                v = float(np.linalg.norm(p - self.prev[name])) / dt
                self.max_speed[name] = max(self.max_speed.get(name, 0.0), v)
            self.prev[name] = p
            self.min_z[name] = min(self.min_z.get(name, np.inf), float(p[2]))

    def explosions(self) -> List[str]:
        bad = [f'{n}: {v:.2f} m/s' for n, v in self.max_speed.items() if v > MAX_SPEED]
        bad += [f'{n}: z {z:.3f}' for n, z in self.min_z.items() if z < TABLE_Z - 0.05]
        return bad


def check_static(env, name: str, level: str, seed: int, idle_s: float = 20.0) -> dict:
    res = {'task': name, 'level': level, 'seed': seed, 'checks': {}, 'notes': []}
    ck = res['checks']
    cls = get_task_class(name)
    # the static task's layout for this seed
    set_level('off')
    te = env.get_task(cls)
    np.random.seed(seed)
    te.reset()
    from pyrep.const import ObjectType
    shapes = [s for s in te._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE) if s.is_respondable()]
    ref = {s.get_name(): np.asarray(s.get_position()) for s in shapes}
    # objects that move by themselves in the STATIC task (e.g. a shape that RLBench spawns where it rolls): their
    # behaviour is the static task's, not the benchmark's, and they are left out of the checks below
    q0 = np.r_[te._task.robot.arm.get_joint_positions(), 1.0]
    for _ in range(int(idle_s / 0.05)):
        te.step(q0)
    unstable = {s.get_name() for s in shapes if np.linalg.norm(np.asarray(s.get_position()) - ref[s.get_name()]) > 0.005}
    unstable |= KNOWN_UNSTABLE.get(cls.BASE_TASK, set())
    if unstable:
        res['notes'].append(f'moving by themselves in the static task too (not checked): {sorted(unstable)}')
    set_level(level)
    te = env.get_task(cls)
    np.random.seed(seed)
    try:
        te.reset()
    except Exception as e:
        ck['placed'] = False
        res['notes'].append(f'placement failed: {type(e).__name__}: {e}')
        return res
    task = te._task
    ck['placed'] = True
    res['layout_rejections'] = len(task.layout_rejections)
    if level in ('L1', 'L2') and not task.layout_rejections:
        diff = [n for n, p in ref.items() if n not in unstable and np.linalg.norm(np.asarray(_pos(task, n)) - p) > 1e-3]
        ck['same_layout'] = not diff
        if diff:
            res['notes'].append(f'layout differs: {diff[:5]}')
    q = np.r_[task.robot.arm.get_joint_positions(), 1.0]
    # inactive: nothing moves
    task.dynbench_set_active(False)
    for _ in range(10):                  # let what was just placed settle (RLBench steps a little before an episode)
        te.step(q)
    before = {n: np.asarray(s.get_position()) for n, s in _bodies(task).items()}
    palm0 = task.person.palm_world() if not task.person.hidden else None
    for _ in range(25):
        te.step(q)
    moved = [n for n, s in _bodies(task).items() if n in before and n not in unstable and
             np.linalg.norm(np.asarray(s.get_position()) - before[n]) > 0.003]
    palm_moved = palm0 is not None and np.linalg.norm(task.person.palm_world() - palm0) > 1e-6
    ck['still_inactive'] = not moved and not palm_moved
    if moved or palm_moved:
        res['notes'].append(f'moved while inactive: {moved} palm={palm_moved}')
    # active, robot still
    task.dynbench_set_active(True)
    watch = Watch(task, ignore=unstable)
    for _ in range(int(idle_s / 0.05)):
        te.step(q)
        watch.step()
    bad = watch.explosions()
    ck['no_explosion'] = not bad
    if bad:
        res['notes'].append(f'explosion: {bad[:4]}')
    unsupported = []
    for it in task.items.values():
        if it.hidden or it.state != 'passive':
            continue
        if np.linalg.norm(it.velocity()) > REST_SPEED or not _supported(task, it):
            unsupported.append(f'{it.name} z={it.position()[2]:.3f} v={np.linalg.norm(it.velocity()):.3f}')
    ck['supported'] = not unsupported
    if unsupported:
        res['notes'].append(f'not at rest / not supported: {unsupported}')
    info = task.dynbench_object_info()
    classes = {e['name']: e['class'] for e in info}
    ok = all(e['class'] in ('static', 'passive', 'agent', 'carried', 'held') for e in info)
    ok &= all(classes.get(p.get_name()) == 'agent' for p in task.person.parts) if not task.person.hidden else True
    ok &= all(classes.get(it.name) in ('passive', 'carried', 'static') for it in task.items.values() if not it.hidden)
    known = {h for e in info for h in e['handles']}
    missing = [h for h in task.dynbench_object_handles() if h not in known]
    ck['oracle'] = bool(ok and not missing)
    if missing:
        res['notes'].append(f'handles without class: {len(missing)}')
    m = task.dynbench_metrics()
    if level in ('L0', 'L1'):
        ok = m.get('obstacle_placed', False)
        if name not in HAND_TASKS:          # (a hand leaves by itself, also at L0)
            ok = ok and m.get('obstacle_in_way_at_end', False)
        ck['obstacle'] = bool(ok)
        if not ok:
            res['notes'].append(f'obstacle not in the way: phase {m.get("script_phase")}')
    # (L2: the obstacle and the traffic come on the robot's progress: with a still robot nothing is expected; the
    # blind and the scripted runs check them)
    if level in ('L1', 'L2'):
        ck['idle_safe'] = not m['collided']
        if m['collided']:
            res['notes'].append(f'contact with a still robot: {m["contact_log"][:2]}')
    res['events'] = [e['event'] for e in m['events']]
    res['person_jobs'] = m['person']['jobs']
    return res


def _pos(task, name):
    from pyrep.objects.object import Object
    return Object.get_object(name).get_position()


def check_blind(env, name: str, level: str, seed: int = 0) -> dict:
    from rlbench.dynamic_bench.keyframes import static_episode
    from rlbench.dynamic_bench.tools.common import replay_open_loop
    res = {'task': name, 'level': level + ' blind', 'seed': seed, 'checks': {}, 'notes': []}
    ck = res['checks']
    set_level(level)
    te = env.get_task(get_task_class(name))
    _, _, demo, kfs = static_episode(te, seed)
    task = te._task
    watch = Watch(task)
    ok, steps, _ = replay_open_loop(te, demo, max_extra_steps=100, on_step=lambda k: watch.step())
    m = task.dynbench_metrics()
    ck['ran'] = True
    # a blind robot may throw what it hits (fast): that is the obstacle mattering, not a physics error
    hit = set(m['items_touched'])
    bad = [b for b in watch.explosions() if b.split(':')[0] not in hit]
    ck['no_explosion'] = not bad
    if bad:
        res['notes'].append(f'explosion: {bad[:4]}')
    gt = task.dynbench_ground_truth()
    ck['ground_truth'] = all(n in gt and len(gt[n]['t']) > 10 for n in [p.get_name() for p in task.person.parts])
    touched = bool(m.get('obstacle_touched_by_robot')) or bool(m['items_touched'])
    # it matters if the blind robot (or what it pushes or holds) touches it or moves it, touches the person, or fails
    moved = (m.get('obstacle_displaced_m') or 0.0) > 0.02
    ck['obstacle_matters'] = bool(touched or moved or m['person_contact_events'] > 0 or not ok)
    if not ck['obstacle_matters']:
        res['notes'].append('the blind static keyframes went through untouched')
    res.update({'success': bool(ok), 'unsafe': m['unsafe'], 'person_contacts': m['person_contact_events'],
                'obstacle_touched': touched, 'obstacle_displaced_m': m.get('obstacle_displaced_m'), 'keyframes': m['keyframes'], 'toppled': m['toppled_by_robot'],
                'events': [e['event'] for e in m['events'] if not e['event'].startswith('person_')]})
    return res


def check_solve(name: str, level: str, seed: int, out: str) -> dict:
    from rlbench.dynamic_bench.tools.solve import run
    res = {'task': name, 'level': level + ' solve', 'seed': seed, 'checks': {}, 'notes': []}
    m = run(name, level, seed, out)
    events = [e['event'] for e in m['events']]
    want = SCENARIOS.get(name, [])
    k = 0
    for e in events:
        if k < len(want) and e == want[k]:
            k += 1
    missing = want[k:] if k < len(want) else []
    if level == 'L2':
        missing += [e for e in SCENARIOS_L2.get(name, []) if e not in events]
    res['checks']['scenario'] = not missing
    if missing:
        res['notes'].append(f'missing events: {missing} (got {[e for e in events if not e.startswith("person_")]})')
    res['checks']['scripted_success'] = bool(m['scripted_success'])
    res['checks']['person_safe'] = m['person_contact_events'] == 0
    if m['person_contact_events']:
        res['notes'].append(f'contacts with the person: {m["contact_log"][:2]}')
    res.update({'unsafe': m['unsafe'], 'person_contacts': m['person_contact_events'], 'keyframes': m['keyframes'],
                'toppled': m['toppled_by_robot'], 'log': m['scripted_log'],
                'extra': {k: m.get(k) for k in m if k not in ('events', 'contact_log', 'scripted_log', 'person')}})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='*', default=list(TASKS))
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--levels', nargs='*', default=['L0', 'L1', 'L2'])
    ap.add_argument('--parts', nargs='*', default=['static', 'blind', 'solve'])
    ap.add_argument('--out', default='')
    ap.add_argument('--shots', default='', help='folder for the scripted remedies\' contact sheets')
    a = ap.parse_args()
    from rlbench.dynamic_bench.tools.common import make_env
    results = []

    def emit(r):
        results.append(r)
        fails = [k for k, v in r['checks'].items() if v is False]
        print(f"{r['task']:34s} {r['level']} s{r['seed']} {'FAIL ' + ','.join(fails) if fails else 'ok'}"
              f"{' | ' + '; '.join(r['notes']) if r['notes'] else ''}", flush=True)
        if a.out:
            with open(a.out, 'w') as f:
                json.dump(results, f, indent=1, default=str)
    if 'static' in a.parts or 'blind' in a.parts:
        env = make_env()
        try:
            for name in a.tasks:
                if 'static' in a.parts:
                    for level in a.levels:
                        for seed in a.seeds:
                            try:
                                emit(check_static(env, name, level, seed))
                            except Exception as e:
                                emit({'task': name, 'level': level, 'seed': seed, 'checks': {'ran': False},
                                      'notes': [f'{type(e).__name__}: {e}', traceback.format_exc(limit=3)]})
                if 'blind' in a.parts:
                    for level in [lv for lv in ('L1', 'L2') if lv in a.levels]:
                        try:
                            emit(check_blind(env, name, level, a.seeds[0]))
                        except Exception as e:
                            emit({'task': name, 'level': level + ' blind', 'seed': a.seeds[0],
                                  'checks': {'ran': False}, 'notes': [f'{type(e).__name__}: {e}']})
        finally:
            env.shutdown()
    if 'solve' in a.parts:
        for name in a.tasks:
            for level in [lv for lv in ('L1', 'L2') if lv in a.levels]:
                try:
                    emit(check_solve(name, level, a.seeds[0], a.shots))
                except Exception as e:
                    emit({'task': name, 'level': level + ' solve', 'seed': a.seeds[0], 'checks': {'ran': False},
                          'notes': [f'{type(e).__name__}: {e}']})
    total = sum(len(r['checks']) for r in results)
    passed = sum(1 for r in results for v in r['checks'].values() if v)
    print(f'\n{passed}/{total} checks passed')


if __name__ == '__main__':
    main()
