"""
Scripted remedies of the benchmark tasks (privileged, cooperative robot: see scripted.py). The robot follows the
STATIC keyframes of the episode and, between two of them, deals with the obstacle in one of the ways the task allows;
the keyframes are never changed. They check that each scenario unfolds as designed and can be solved:
    python -m rlbench.dynamic_bench.tools.solve dyn_close_jar_item_in_path --level L1 --seed 3 --out /tmp/solve

Remedies shown (one per task; the tasks list the others):
    stack_blocks      park the held block, pick the block off the spot, (L2) unstack the parked one, re-grasp
    close_jar         carry the lid over the item (a detour above it, then the keyframe)
    shape_sorter      park the shape, drag the tablet off the hole (pressing on it), re-grasp the shape;
                      topple the box off the shape (an arc push about its bottom edge), then grasp
    wine              stand the own bottle up, move the person's bottle by the neck, re-grasp the own one
    groceries         pick the item away (or push it), then grasp as decided
    slide_block       stop the push, take the obstacle away (pick or push) or turn the board about its held end
                      (an arc push), resume the push
    insert_peg        wait for the hand to leave
"""

import argparse
import json
import os
from typing import Callable, Dict, List

import numpy as np

from rlbench.dynamic_bench import get_task_class, set_level
from rlbench.dynamic_bench.assets import _mesh_world, quat_to_matrix
from rlbench.dynamic_bench.core import TABLE_Z, world_aabb
from rlbench.dynamic_bench.tools.scripted import ScriptedRobot, Timeout, free_table_spot, shape_yaw

SOLVERS: Dict[str, Callable] = {}
GRASPABLE = 0.07         # [m] widest horizontal extent the gripper closes on


def solver(name):
    """Registers a remedy; every remedy first lets the person finish what she is bringing, out of her way."""
    def deco(fn):
        def run_remedy(rb, task, kfs, rng):
            rb.hold(0.3)
            j = task.person.job
            if task.phase == 'coming' or (j is not None and j.name != 'rest'):
                settle(rb, task)
            return fn(rb, task, kfs, rng)
        SOLVERS[name] = run_remedy
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
    from rlbench.dynamic_bench.core import robot_reachable
    q = rb.arm.get_tip().get_quaternion()
    for _ in range(4):
        # (by default the near part of the table: a spot the robot surely takes again; the far edge is out of reach)
        kw = {'region': ((0.10, 0.36), (-0.30, 0.30)) if prefer is None else ((0.10, 0.45), (-0.32, 0.32))} \
            if region is None else {'region': region}
        # where the robot can take it again (within its usual reach), unless the caller knows better (stack_blocks
        # parks where the person can reach; the robot still re-grasps a block there)
        kw['prefer'] = prefer if prefer is not None else robot_reachable
        spot = free_table_spot(task, radius, avoid, rng, **kw)
        if spot is None:
            return False
        if rb.place(spot + [0, 0, height], 0.0, quat=q):
            return True
        avoid = list(avoid) + [spot]
        rb.retreat(0.06)                         # (the start may touch something: a little higher)
    return False


def settle(rb: ScriptedRobot, task, max_seconds: float = 15.0) -> None:
    """Do not act on an obstacle while it is still being brought (privileged: the script phase); out of the
    person's way meanwhile."""
    rb.hold(0.3)
    j = task.person.job
    rb.make_room(always=task.phase == 'coming' or (j is not None and j.name != 'rest'))
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
        # (and away from the robot's base and from where the robot waits: she must not have to reach past it)
        reach = (lambda xy: p.reach_lean(p.palm_above(TABLE_Z + 0.11, xy)) is not None and
                 np.linalg.norm(xy - task.goal[:2]) > 0.15 and
                 np.linalg.norm(xy - np.array([-0.31, 0.0])) > 0.45 and
                 np.linalg.norm(xy - rb.safe_pose()[:2]) > 0.30)
        if not park(rb, task, 0.5 * 0.05 + 0.004, [task.goal], rng, prefer=reach):
            return False
        # step back and let the person do what she does with the parked block (L2) before moving on
        rb.make_room(always=True)
        rb.hold(1.5)
        rb.hold(until=lambda: getattr(task, 'park_phase', 'none') not in ('parked', 'second_coming') and not task.person.busy,
                max_seconds=25.0)
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
        if not _pick_block(rb, held):
            return False
        from rlbench.dynamic_bench.core import ROBOT_BASE_XY, robot_reachable
        if np.linalg.norm(np.asarray(held.get_position())[:2] - ROBOT_BASE_XY) > 0.66:
            # parked where she can reach, at the edge of the robot's reach: the grasp there is not precise enough
            # for a tower; put it down nearer and take it again
            near = free_table_spot(task, 0.045, [task.goal], rng, region=((0.05, 0.30), (-0.28, 0.28)),
                                   prefer=lambda xy: robot_reachable(xy, 0.10))
            if near is not None and rb.place(near + [0, 0, 0.5 * 0.05 + 0.004], shape_yaw(held)):
                rb.log('  regrasp nearer, for a precise grasp')
                return _pick_block(rb, held)
        return True
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


def _sorter_grasp_point(task):
    from pyrep.objects.dummy import Dummy
    from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES
    return Dummy(SHAPE_NAMES[task.variation].replace(' ', '_') + '_grasp_point')


@solver('dyn_shape_sorter_tablet_on_hole')
def solve_sorter(rb: ScriptedRobot, task, kfs, rng) -> None:
    from rlbench.dynamic_bench.core import robot_reachable
    closes, opens = transitions(kfs)
    above_hole = opens[0] - 1
    gp = _sorter_grasp_point(task)
    shape = task.shape

    def before(i):
        if i != above_hole:
            return True
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        rb.log('  the tablet covers the hole: park the shape, drag the tablet off (pressing on it), re-grasp')
        h = gp.get_position()[2] - world_aabb(shape)[0][2]
        if not park(rb, task, h + 0.004, [task.hole], rng, radius=0.04):
            rb.log('  no way to park the shape')
            return False
        rb.wait_person()
        tb = task.tablet
        f = tb.frame()
        c = f[:2, 3]
        # along a side of the sorter: the shortest drag that uncovers the hole, the tablet's centre staying on the
        # sorter (it would tip off), where the gripper can press and end
        options = []
        for k in range(4):
            a = task.sorter_yaw + k * np.pi / 2
            d = np.array([np.cos(a), np.sin(a)])
            h_d = abs(np.dot(f[:2, 0], d)) * tb.half_extents[0] + abs(np.dot(f[:2, 1], d)) * tb.half_extents[1]
            dist = h_d + float(np.dot(task.hole[:2] - c, d)) + 0.06       # its edge well clear of the hole
            end_c = c + d * dist
            on = abs(float(np.dot(end_c - task.sorter_xy, d))) < task.sorter_half + 0.012   # (the plate's edge)
            if robot_reachable(c, 0.04) and robot_reachable(end_c, 0.04):
                options.append((not on, dist, d))       # where it stays on the sorter first, then the shortest
        options.sort(key=lambda o: (o[0], o[1]))
        if not options:
            rb.log('    no way to drag it off the hole')
        for _, dist, d in options[:2]:
            rb.log(f'    drag from {np.round(c, 3)} along {np.round(d, 2)} by {dist:.3f} m')
            rb.drag(c, tb.top_z(), d, dist)
            rb.hold(0.5)
            if not task.obstacle_in_way():
                break
            c = tb.frame()[:2, 3]
        return rb.pick(shape, gp.get_position(), 0.0, quat=gp.get_quaternion())
    rb.follow_keyframes(kfs, before=before)
    rb.hold(1.5)


@solver('dyn_shape_sorter_box_over_shape')
def solve_box(rb: ScriptedRobot, task, kfs, rng) -> None:
    from rlbench.dynamic_bench.core import ROBOT_BASE_XY
    closes, opens = transitions(kfs)
    above_shape = closes[0] - 1
    gp = _sorter_grasp_point(task)

    def before(i):
        if i not in (above_shape, closes[0]):
            return True
        if task.is_grasped(task.shape):
            return 'skip'                      # (grasped by the remedy: the approach and the grasp are done)
        settle(rb, task)
        if not task.obstacle_in_way():
            return True
        if not topple():
            return False
        # the shape may have turned a little as the box went over it: grasp it as the demo did (the grasp point
        # moves with it), so that the next keyframes bring it to the hole aligned
        return 'skip' if rb.pick(task.shape, gp.get_position(), 0.0, quat=gp.get_quaternion()) else False

    def topple() -> bool:
        bx = task.box
        hw = float(bx.half_extents[0])
        rb.log('  the box covers the shape: topple it (push high on one side, following its edge)')
        for _ in range(2):
            c = bx.position()[:2]
            dirs = [d for d in task.box_axes(bx.yaw()) if task.topple_ok(c, d)]
            # not where the open fingers of the grasp keyframe come down (the box lies down on that side), then
            # away from the robot
            fa = quat_to_matrix(kfs[closes[0]]['quaternion'])[:2, 1]
            fa = fa / max(np.linalg.norm(fa), 1e-9)
            away = (c - ROBOT_BASE_XY) / np.linalg.norm(c - ROBOT_BASE_XY)
            dirs.sort(key=lambda d: (round(abs(float(np.dot(d, fa))), 1), -float(np.dot(d, away))))
            for d in dirs:
                hc = 0.7 * bx.height
                pivot = np.r_[c + d * hw, TABLE_Z]
                start = np.r_[c - d * (hw + 0.012), TABLE_Z + hc]
                axis = np.cross([0.0, 0.0, 1.0], np.r_[d, 0.0])
                rb.log(f'    topple along {np.round(d, 2)} from {np.round(start, 3)}')
                rb.arc_push(pivot, start, axis, np.radians(60), float(np.arctan2(d[1], d[0])) + np.pi / 2)
                rb.hold(1.0)
                if not task.obstacle_in_way():
                    return True
        return not task.obstacle_in_way()
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


def _slide(rb: ScriptedRobot, task, kfs, clear_obstacle: Callable[[], bool]) -> None:
    """Pushes the block along the static keyframes; stops while the obstacle is in the way (or being brought),
    clears it (clear_obstacle), goes back behind the block and pushes on."""
    push = len(kfs) - 1                         # behind the block -> pushed onto the target
    start = np.asarray(kfs[push - 1]['position'])
    rel = start[:2] - task.c0[:2]               # the tip behind the block, as decided

    def clear() -> bool:
        rb.go_safe()                            # out of the person's way while she brings it
        settle(rb, task)
        rb.wait_person()
        if not task.obstacle_in_way():
            return True
        return clear_obstacle()

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


@solver('dyn_slide_block_obstacle_in_path')
def solve_slide(rb: ScriptedRobot, task, kfs, rng) -> None:
    def take_away() -> bool:
        it = task.obstacle
        rb.log(f'  the {task.obstacle_kind} is in the block\'s way: take it away')
        if task.obstacle_kind == 'mug' and pick_away(rb, task, it.shape, rng, [task.c0, task.end]):
            return True
        d = away_direction(task, it.position(), along=task.u, avoid=[task.end])
        return push_off(rb, task, it.shape, d, 0.02)
    _slide(rb, task, kfs, take_away)


@solver('dyn_slide_block_board_pinned')
def solve_board(rb: ScriptedRobot, task, kfs, rng) -> None:
    def rotate() -> bool:
        bd = task.board
        for _ in range(2):
            pivot = task.pivot()
            f = bd.frame()
            axis_b = f[:2, 0]
            c = f[:2, 3]
            # the free end: the end away from the pivot
            free = c + axis_b * bd.half_extents[0] * (1.0 if np.dot(c - pivot, axis_b) > 0 else -1.0)
            r_dir = (free - pivot) / np.linalg.norm(free - pivot)
            r = float(np.linalg.norm(free - pivot)) - 0.03
            e = task.swing if task.swing is not None else task.u
            e = e - r_dir * np.dot(e, r_dir)                        # perpendicular to the board
            e = e / np.linalg.norm(e)
            contact = pivot + r_dir * r - e * (bd.half_extents[1] + 0.012)
            sgn = 1.0 if np.cross(r_dir, e) > 0 else -1.0         # turning about +z moves r_dir towards e if > 0
            rb.log(f'  the board is across the block\'s way, held by the book: turn it about {np.round(pivot, 3)}, '
                   f'its free end along {np.round(e, 2)}')
            rb.arc_push(np.r_[pivot, TABLE_Z], np.r_[contact, TABLE_Z + 0.009], [0.0, 0.0, sgn], np.radians(85),
                        float(np.arctan2(e[1], e[0])) + np.pi / 2)
            rb.hold(0.5)
            if not task.obstacle_in_way():
                return True
        return not task.obstacle_in_way()
    _slide(rb, task, kfs, rotate)


@solver('dyn_stack_blocks_tunnel')
def solve_tunnel(rb: ScriptedRobot, task, kfs, rng) -> None:
    from rlbench.dynamic_bench.tasks.dyn_stack_blocks_tunnel import push_ok
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
        rb.log(f'  the tunnel covers the block: slide it off along its opening, pushing the roof (tunnel at '
               f'{np.round(c, 3)}, opening {np.round(u, 2)}, ends reachable '
               f'{[bool(push_ok(c, d)) for d in (u, -u)]})')
        # from the end the gripper can get to, just far enough for the block to come out of it
        for dd in sorted((u, -u), key=lambda d: -float(push_ok(c, d))):
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
        rb.hold(1.0)
        if task.obstacle_in_way() or task.person.busy:
            rb.log('  the hand is on the spoke: wait for it to leave, out of its way')
            rb.make_room(always=True)
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
