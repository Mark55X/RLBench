"""
Scripted remedies of the benchmark tasks (privileged, cooperative robot: see scripted.py). The robot follows the
STATIC keyframes of the episode and, between two of them, deals with the obstacle in one of the ways the task allows;
the keyframes are never changed. They check that each scenario unfolds as designed and can be solved:
    python -m rlbench.dynamic_bench.tools.solve dyn_close_jar_item_in_path --level L1 --seed 3 --out /tmp/solve

Remedies shown (one per task; the tasks list the others):
    stack_blocks      park the held block, pick the block off the spot, (L2) unstack the parked one, re-grasp
    close_jar         carry the lid over the item (a detour above it, then the keyframe)
    shape_sorter      park the shape, push the book off the hole, re-grasp the shape
    wine              stand the own bottle up, move the person's bottle by the neck, re-grasp the own one
    groceries         pick the item away (or push it), then grasp as decided
    slide_block       stop the push, take the obstacle away (pick or push), resume the push
    insert_peg        wait for the hand to leave
"""

import argparse
import json
import os
from typing import Callable, Dict, List

import numpy as np

from rlbench.dynamic_bench import get_task_class, set_level
from rlbench.dynamic_bench.assets import _mesh_world
from rlbench.dynamic_bench.core import TABLE_Z, world_aabb
from rlbench.dynamic_bench.tools.scripted import ScriptedRobot, Timeout, free_table_spot, shape_yaw

SOLVERS: Dict[str, Callable] = {}
GRASPABLE = 0.07         # [m] widest horizontal extent the gripper closes on


def solver(name):
    def deco(fn):
        SOLVERS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def transitions(kfs: List[dict]):
    """Indices of the keyframes where the gripper closes (grasps) and where it opens (releases)."""
    closes, opens = [], []
    prev = 1.0
    for i, k in enumerate(kfs):
        if prev > 0.5 and k['gripper_open'] < 0.5:
            closes.append(i)
        if prev < 0.5 and k['gripper_open'] > 0.5:
            opens.append(i)
        prev = k['gripper_open']
    return closes, opens


def narrow(shape):
    """(finger yaw, width): the horizontal direction across which the shape is narrowest (from its mesh) and the
    width along it."""
    v = _mesh_world(shape)[:, :2]
    v = v - v.mean(0)
    best = None
    for a in np.linspace(0.0, np.pi, 36, endpoint=False):
        d = np.array([np.cos(a), np.sin(a)])
        w = float(np.ptp(v @ d))
        if best is None or w < best[1]:
            best = (float(a), w)
    return best


def park(rb: ScriptedRobot, task, height: float, avoid, rng, radius: float = 0.045, region=None, prefer=None) -> bool:
    """Puts what the gripper holds down on a free table spot (tip height above the table), keeping the gripper's
    orientation; tries a few spots."""
    q = rb.arm.get_tip().get_quaternion()
    for _ in range(4):
        kw = {} if region is None else {'region': region}
        if prefer is not None:
            kw['prefer'] = prefer
        spot = free_table_spot(task, radius, avoid, rng, **kw)
        if spot is None:
            return False
        if rb.place(spot + [0, 0, height], 0.0, quat=q):
            return True
        avoid = list(avoid) + [spot]
    return False


def settle(rb: ScriptedRobot, task, max_seconds: float = 15.0) -> None:
    """Do not act on an obstacle while it is still being brought (privileged: the script phase); out of the
    person's way meanwhile."""
    rb.hold(0.3)
    rb.make_room()
    rb.hold(until=lambda: task.phase != 'coming' and not (task.person.job is not None and
                                                          task.person.job.name not in ('rest',)),
            max_seconds=max_seconds)


def pick_away(rb: ScriptedRobot, task, shape, rng, avoid, top=None) -> bool:
    """Picks the object top-down across its narrow side and puts it on a free spot of the table."""
    yaw, width = narrow(shape)
    lo, hi = world_aabb(shape)
    c = 0.5 * (lo + hi)
    hgt = hi[2] - lo[2]
    g = np.array([c[0], c[1], (hi[2] if top is None else top) - min(0.03, 0.5 * hgt)])
    if width > GRASPABLE or not rb.pick(shape, g, yaw):
        return False
    spot = free_table_spot(task, 0.06, avoid, rng)
    return spot is not None and rb.place(spot + [0, 0, g[2] - lo[2] + 0.005], yaw)


def push_off(rb: ScriptedRobot, task, shape, direction, height: float, extra: float = 0.06) -> bool:
    """Pushes the object along direction by its extent plus a margin, the closed gripper at height above the
    surface the object stands on."""
    d = np.asarray(direction, dtype=np.float64)[:2]
    d = d / np.linalg.norm(d)
    lo3, hi3 = world_aabb(shape)
    height = min(height, 0.5 * float(hi3[2] - lo3[2]))      # a flat object is pushed low (the fingers end at the tip)
    v = _mesh_world(shape)[:, :2]
    c = 0.5 * (v.min(0) + v.max(0))
    back = float(np.max((v - c) @ -d))
    ext = float(np.ptp(v @ d))
    lo = world_aabb(shape)[0]
    start = np.r_[c - d * (back + 0.035), 0.0]
    for sign in (1.0, -1.0):                  # the other way if the gripper cannot get behind it
        dd = sign * d
        back = float(np.max((v - c) @ -dd))
        start = np.r_[c - dd * (back + 0.035), 0.0]
        rb.log(f'    push from {np.round(start[:2], 3)} along {np.round(dd, 2)} by {ext + extra:.2f} m, '
               f'{(lo[2] - TABLE_Z) + height:.3f} m above the table')
        if rb.push(start, np.r_[dd, 0.0], ext + extra, (lo[2] - TABLE_Z) + height):
            return True
    return False


def away_direction(task, at, along=None, avoid=()) -> np.ndarray:
    """A horizontal direction to push something at 'at' that leads to free table (perpendicular to along if
    given), away from the points in avoid."""
    at = np.asarray(at, dtype=np.float64)[:2]
    cands = []
    if along is not None:
        u = np.asarray(along, dtype=np.float64)[:2]
        u = u / np.linalg.norm(u)
        cands = [np.array([-u[1], u[0]]), np.array([u[1], -u[0]])]
    else:
        cands = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]

    def score(d):
        land = at + d * 0.15
        s = -abs(land[1]) - 0.5 * abs(land[0] - 0.25)
        for q in avoid:
            s -= max(0.0, 0.2 - np.linalg.norm(land - np.asarray(q)[:2])) * 5
        return s
    return max(cands, key=score)


# ---------------------------------------------------------------------------------------------------------------------
# solvers: (robot, task, static keyframes, rng)
# ---------------------------------------------------------------------------------------------------------------------

def _pick_block(rb: ScriptedRobot, b) -> bool:
    yaw = shape_yaw(b)
    return rb.pick(b, b.get_position(), yaw, alt_yaws=(yaw + np.pi / 2,))


def _block_on(task, base):
    b = np.asarray(base.get_position())
    for s in list(task.target_blocks) + list(task.distractors):
        if s.get_handle() == base.get_handle():
            continue
        p = np.asarray(s.get_position())
        if np.linalg.norm(p[:2] - b[:2]) < 0.03 and p[2] > b[2] + 0.03:
            return s
    return None


@solver('dyn_stack_blocks_goal_taken')
def solve_stack(rb: ScriptedRobot, task, kfs, rng) -> None:
    closes, opens = transitions(kfs)
    above_spot = opens[0] - 1                 # above the target plane, the first block in the hand

    def before(i):
        if i != above_spot:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        held = task._held_target()
        if held is None:
            rb.log('  the block slipped: take it again')
            held = task.first
            if not _pick_block(rb, held):
                return False
        rb.log('  the stacking spot is taken: park the held block, take the other block off the spot')
        p = task.person
        # within the person's reach (so that, at L2, she can do what she does with a parked block)
        reach = (lambda xy: p.reach_lean(p.palm_above(TABLE_Z + 0.11, xy)) is not None and
                 np.linalg.norm(xy - task.goal[:2]) > 0.15)
        if not park(rb, task, 0.5 * 0.05 + 0.004, [task.goal], rng, prefer=reach):
            return False
        rb.wait_person()
        blk = task.obstacle
        spot2 = free_table_spot(task, 0.045, [task.goal, held.get_position()], rng)
        if not _pick_block(rb, blk.shape) or spot2 is None or \
                not rb.place(spot2 + [0, 0, 0.5 * 0.05 + 0.004], shape_yaw(blk.shape)):
            return False
        rb.wait_person()
        rb.hold(1.0)
        rb.wait_person()
        top = _block_on(task, held)
        if top is not None:
            rb.log(f'  a block was stacked on the parked one: lift it off')
            spot3 = free_table_spot(task, 0.045, [task.goal, held.get_position()], rng)
            if not _pick_block(rb, top) or spot3 is None or \
                    not rb.place(spot3 + [0, 0, 0.5 * 0.05 + 0.004], shape_yaw(top)):
                return False
        rb.log('  re-grasp the parked block, back to the keyframes')
        return _pick_block(rb, held)
    rb.follow_keyframes(kfs, before=before)
    rb.hold(2.0)


@solver('dyn_close_jar_item_in_path')
def solve_close_jar(rb: ScriptedRobot, task, kfs, rng) -> None:
    closes, opens = transitions(kfs)
    above_jar = closes[0] + 2                 # grasp, lift, above the jar
    rb.room_always = True                     # the lid is right in front of the person: out of her way when waiting
    rb.hold(0.3)
    settle(rb, task)                          # she is bringing something across the table: let her finish first

    def before(i):
        if i != above_jar:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        it = task.obstacle
        z = it.top_z() + 0.07                 # the lid's bottom 5 cm over the item
        kf = kfs[i]
        q = kf['quaternion']
        tip = rb.tip()
        rb.log(f'  the {task.obstacle_kind} is in the way: carry the lid over it (detour at {z:.2f} m)')
        ok = rb.move([tip[0], tip[1], z], q, linear=True, ignore_collisions=True)
        ok = ok and rb.move([kf['position'][0], kf['position'][1], z], q, linear=True, ignore_collisions=True,
                            steps=60)
        return ok
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_shape_sorter_book_on_hole')
def solve_sorter(rb: ScriptedRobot, task, kfs, rng) -> None:
    from pyrep.objects.dummy import Dummy
    from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES
    closes, opens = transitions(kfs)
    above_hole = opens[0] - 1
    gp = Dummy(SHAPE_NAMES[task.variation].replace(' ', '_') + '_grasp_point')
    shape = task.shape

    def before(i):
        if i != above_hole:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        rb.log('  the book covers the hole: park the shape, slide the book off, re-grasp the shape')
        h = gp.get_position()[2] - world_aabb(shape)[0][2]
        parked = park(rb, task, h + 0.004, [task.hole], rng, radius=0.04)
        if not parked:
            rb.log('  no way to park the shape: push the book off with the shape in the hand')
        rb.wait_person()
        book = task.book
        from rlbench.dynamic_bench.core import robot_reachable
        c = book.position()[:2]
        # off the sorter (it drops flat), along a side of it, where the gripper can start and end the push
        sides = [np.array([np.cos(a), np.sin(a)]) for a in np.arange(4) * np.pi / 2 + task.sorter_yaw]
        sides.sort(key=lambda d: -(robot_reachable(c - d * 0.15, 0.03) + robot_reachable(c + d * 0.25, 0.03)))
        for d in sides:
            if push_off(rb, task, book.shape, d, 0.008, extra=0.12):
                break
        if not parked:
            return rb.is_holding(shape) if hasattr(rb, 'is_holding') else task.is_grasped(shape)
        return rb.pick(shape, gp.get_position(), 0.0, quat=gp.get_quaternion())
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_place_wine_bottle_in_path')
def solve_wine(rb: ScriptedRobot, task, kfs, rng) -> None:
    closes, opens = transitions(kfs)
    grasp = closes[0]
    in_front = opens[0] - 1
    own = task.wine_bottle
    offset = None
    half_h = 0.5 * float(world_aabb(own)[1][2] - world_aabb(own)[0][2])      # standing

    def before(i):
        nonlocal offset
        if i == grasp + 1:
            offset = np.asarray(kfs[grasp]['position']) - np.asarray(own.get_position())    # tip vs bottle centre
        if i != in_front:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        rb.log('  the person\'s bottle stands in the way: stand the own bottle on the table, move hers by the neck')
        if not park(rb, task, half_h + offset[2] + 0.005, [task.bottle_pos], rng, radius=0.05,
                    region=((0.05, 0.40), (-0.30, 0.15))):
            return False
        rb.wait_person()
        other = task.bottle
        neck = other.position() + [0, 0, other.height - 0.035]
        spot2 = free_table_spot(task, 0.05, [task.bottle_pos, own.get_position(), task.w_in], rng,
                                region=((0.05, 0.40), (-0.30, 0.10)))
        if spot2 is None or not rb.pick(other.shape, neck, 0.0) or \
                not rb.place(spot2 + [0, 0, other.height - 0.035 + 0.005], 0.0):
            return False
        rb.log('  re-grasp the own bottle as decided (from the side, as the grasp keyframe)')
        g = np.asarray(own.get_position()) + offset
        approach = np.asarray(kfs[grasp - 1]['position']) - np.asarray(kfs[grasp]['position'])
        return rb.pick_along(own, g, kfs[grasp]['quaternion'], approach, lift=0.03)
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_put_groceries_grasp_blocked')
def solve_groceries(rb: ScriptedRobot, task, kfs, rng) -> None:
    def before(i):
        if i != 0:
            return True
        # come over the target (the L2 item is brought then), let the person finish
        # wait 31 cm above the grasp keyframe (the person brings the L2 item when the gripper comes this close)
        w = task.w_grasp
        rb.move([w[0], w[1], w[2] + 0.31], kfs[0]['quaternion'])
        rb.hold(0.5)
        settle(rb, task)
        rb.wait_person()
        for o, pos, side in task.blockers:
            still = o.state == 'passive' and np.linalg.norm(o.position()[:2] - pos[:2]) < 0.025
            if task.mode == 'on_top':
                still = task.obstacle_in_way()
            if not still:
                continue
            rb.log(f'  {o.name} is in the way of the grasp: take it away')
            if pick_away(rb, task, o.shape, rng, [task.tgt.position()]):
                continue
            rb.log('    too wide to grasp: push it')
            if task.mode == 'on_top':
                d = away_direction(task, o.position())
                push_off(rb, task, o.shape, d, 0.01)
            else:
                d = away_direction(task, o.position(), along=side, avoid=[task.tgt.position()])
                push_off(rb, task, o.shape, d, 0.03)
        return True
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_slide_block_obstacle_in_path')
def solve_slide(rb: ScriptedRobot, task, kfs, rng) -> None:
    push = len(kfs) - 1                         # behind the block -> pushed onto the target
    start = np.asarray(kfs[push - 1]['position'])
    rel = start[:2] - task.c0[:2]               # the tip behind the block, as decided

    def clear() -> bool:
        rb.go_safe()                            # out of the person's way while she brings it
        settle(rb, task)
        rb.wait_person()
        if not task.obstacle_in_way():
            return True
        it = task.obstacle
        rb.log(f'  the {task.obstacle_kind} is in the block\'s way: take it away')
        if task.obstacle_kind == 'mug' and pick_away(rb, task, it.shape, rng, [task.c0, task.end]):
            return True
        d = away_direction(task, it.position(), along=task.u, avoid=[task.end])
        return push_off(rb, task, it.shape, d, 0.02)

    if not rb.follow_keyframes(kfs, stop=push):
        return
    q = kfs[push]['quaternion']
    goal = np.asarray(kfs[push]['position'])
    def busy():                                # the obstacle is in the way, or the person is moving it
        j = task.person.job
        return task.obstacle_in_way() or task.phase == 'coming' or (j is not None and j.name not in ('rest',))

    for _ in range(4):
        if busy():
            rb.retreat(0.12)
            if not clear():
                return
            b = np.asarray(task.block.get_position())
            back = np.r_[b[:2] + rel, start[2]]
            rb.set_gripper(False)
            rb.move(back + [0, 0, 0.10], q)
            rb.move(back, q, linear=True, ignore_collisions=True)
        # push in short steps (stop as soon as something comes into the way)
        tip = rb.tip()
        rest = goal - tip
        n = max(1, int(np.ceil(np.linalg.norm(rest[:2]) / 0.03)))
        stopped = False
        for k in range(1, n + 1):
            if busy():
                stopped = True
                break
            rb.move(tip + rest * k / n, q, linear=True, ignore_collisions=True, steps=8)
        if not stopped:
            break
    rb.hold(1.5)


SOLVERS['dyn_slide_block_target_taken'] = solve_slide
SOLVERS['dyn_slide_block_book_across'] = solve_slide


@solver('dyn_stack_blocks_tunnel')
def solve_tunnel(rb: ScriptedRobot, task, kfs, rng) -> None:
    from rlbench.dynamic_bench.core import robot_reachable
    closes, opens = transitions(kfs)
    above = closes[task.covered_index] - 1   # above the block the tunnel covers

    def before(i):
        if i != above:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        tn = task.tunnel
        u = tn.frame()[:2, 0]                 # the tunnel's opening
        c = tn.position()[:2]
        rb.log('  the tunnel covers the block: slide it off along its opening, pushing the roof')
        # from the end the gripper can get to, just far enough for the block to come out of it
        for dd in sorted((u, -u), key=lambda d: -float(robot_reachable(c - d * 0.115, 0.05))):
            start = np.r_[c - dd * (0.08 + 0.035), 0.0]
            if rb.push(start, np.r_[dd, 0.0], 0.20, tn.height - 0.010, steps=100):
                rb.hold(0.5)
                return True
        return False
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_insert_peg_hand_in_way')
def solve_peg(rb: ScriptedRobot, task, kfs, rng) -> None:
    closes, opens = transitions(kfs)
    above = opens[0] - 1

    def before(i):
        if i != above:
            return True
        if task.level == 'L0':
            rb.log('  the hand rests next to the spoke for good: insert carefully, from above')
            return True
        rb.hold(1.0)
        if task.obstacle_in_way() or task.person.busy:
            rb.log('  the hand is at the spoke: wait for it to leave')
        rb.hold(until=lambda: not task.obstacle_in_way() and not (task.person.busy and task.person.job is not None
                                                                  and task.person.job.name != 'rest'),
                max_seconds=30.0)
        return True
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


# ---------------------------------------------------------------------------------------------------------------------

def run(name: str, level: str, seed: int, out: str, max_steps: int = 6000) -> dict:
    from rlbench.dynamic_bench.keyframes import static_episode
    from rlbench.dynamic_bench.tools.common import SnapshotCamera, make_env, montage, save_png
    set_level(level)
    env = make_env()
    try:
        te = env.get_task(get_task_class(name))
        _, _, _, kfs = static_episode(te, seed)
        task = te._task
        lines = []
        rb = ScriptedRobot(te, max_steps=max_steps, log=lambda s: (lines.append(s), print(s, flush=True)))
        side = task.person.side            # the camera looks from the other side of the table than the person
        cam = SnapshotCamera(eye=(1.30, -0.95 * side, 1.65), look_at=(0.22, 0.0, 0.82), resolution=(480, 360),
                             fov_deg=55) if out else None
        shots = []
        orig_act = rb.act

        def act(q):
            orig_act(q)
            if cam is not None and rb.steps % 20 == 0:
                shots.append(cam.capture())
        rb.act = act
        rb.log('static keyframes: ' + '; '.join(
            f'{k + 1} {np.round(kf["position"], 3).tolist()} {"open" if kf["gripper_open"] > 0.5 else "closed"}'
            for k, kf in enumerate(kfs)))
        error = None
        try:
            SOLVERS[name](rb, task, kfs, np.random.default_rng(seed))
        except Timeout as e:
            error = f'timeout {e}'
        m = task.dynbench_metrics()
        m['scripted_success'] = bool(rb.success or task.success()[0])
        m['scripted_steps'] = rb.steps
        m['scripted_error'] = error
        m['scripted_log'] = lines
        if out:
            os.makedirs(out, exist_ok=True)
            if shots:
                stride = max(1, int(np.ceil(len(shots) / 48)))
                sel = list(range(0, len(shots), stride))
                save_png(montage([shots[k] for k in sel], [f'{k * 20}' for k in sel], cols=8),
                         os.path.join(out, f'solve_{name}_{level}_{seed}.png'))
            with open(os.path.join(out, f'solve_{name}_{level}_{seed}.json'), 'w') as f:
                json.dump(m, f, indent=1, default=str)
        return m
    finally:
        env.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('--level', default='L1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    m = run(a.task, a.level, a.seed, a.out)
    for e in m['events']:
        if e['event'].startswith('person_') and e['event'] not in ('person_flinches', 'person_gives_up'):
            continue
        print('  %6.2f %-26s %s' % (e['t'], e['event'], {k: v for k, v in e.items() if k not in ('t', 'event', 'tip')}))
    keys = ['scripted_success', 'scripted_error', 'keyframes', 'collided', 'person_contact_events',
            'toppled_by_robot', 'fell_off_by_robot', 'unsafe', 'obstacle', 'obstacle_cleared', 'script_phase']
    print(json.dumps({k: m.get(k) for k in keys}, default=str))


if __name__ == '__main__':
    main()
