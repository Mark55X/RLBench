"""
Simulator tests of the benchmark building blocks (CoppeliaSim, headless):
    python -m rlbench.dynamic_bench.tests.test_sim
- items: every library object stands on the table where set_pose puts it (bottom on the table, frame recovered),
  realistic masses, the cover cup is hollow (a block fits under it);
- person: the arm reaches what is within reach (palm on target), reports what is not, carries an item with the palm,
  flinches when the robot touches it, puts back what she holds when a job is abandoned, hides / shows;
- static keyframes: extracted from the static demo of the episode, tracked along the episode;
- monitor: a contact with the person is a collision, a contact with an item is not;
- activation, demo mode, levels.
"""

import os
import sys

import numpy as np

from rlbench.dynamic_bench import assets, core, get_task_class, set_level
from rlbench.dynamic_bench.assets import Item, _mesh_world
from rlbench.dynamic_bench.core import TABLE_Z
from rlbench.dynamic_bench.person import Hold, Job, Move
from rlbench.dynamic_bench.tools.common import make_env, record_static_demo

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def test_items(env):
    pr = env._pyrep
    for kind in assets.SPECS:
        it = Item(pr, kind, 't_' + kind)
        it.place([0.3, 0.1, TABLE_Z], 0.7, passive=False)
        v = _mesh_world(it.shape)
        frame_ok = np.allclose(it.position(), [0.3, 0.1, TABLE_Z], atol=1e-4) and abs(it.yaw() - 0.7) < 1e-3
        check(frame_ok and abs(v[:, 2].min() - TABLE_Z) < 1e-3 and it.tilt() < 1e-3,
              f'{kind}: stands on the table where set_pose puts it (h={it.height:.3f} m, {it.mass} kg)')
        it.remove()
    # the cover cup is hollow: a 5 cm block under it does not intersect it
    from pyrep.const import PrimitiveShape
    from pyrep.objects.shape import Shape
    block = Shape.create(PrimitiveShape.CUBOID, [0.05, 0.05, 0.05], position=[0.3, 0.1, TABLE_Z + 0.026],
                         static=True, respondable=True)
    cup = Item(pr, 'cover_cup', 't_cover')
    hits = []
    for yaw in np.linspace(0, np.pi / 2, 7):
        block.set_orientation([0, 0, yaw])
        cup.place([0.3, 0.1, TABLE_Z + 0.001], 0.3, passive=False)
        hits.append(cup.shape.check_collision(block))
    check(not any(hits), 'cover cup: a 5 cm block fits under it at any yaw')
    cup.place([0.3, 0.1, TABLE_Z + 0.001], 0.3, passive=False)
    block.set_position([0.3, 0.1, TABLE_Z + 0.15])
    check(cup.shape.check_collision(block), 'cover cup: its closed top is solid')
    block.remove()
    cup.remove()


def test_person(env):
    set_level('L0')                    # (at L1 / L2 the person may start with the obstacle in her hand)
    te = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
    np.random.seed(0)
    te.reset()
    task = te._task
    task.dynbench_set_active(False)
    p = task.person
    p.place(-1, 0.3)
    near = [0.30, -0.35, TABLE_Z + 0.05]
    p.set_palm(near)
    check(np.linalg.norm(p.palm_world() - near) < 1e-3 and p.lean < np.radians(5), 'person: near point reached upright')
    far = [0.45, -0.10, TABLE_Z + 0.05]
    lean = p.reach_lean(far)
    p.set_palm(far)
    check(lean is not None and np.linalg.norm(p.palm_world() - far) < 1e-3 and p.lean > 0.05,
          'person: farther point reached leaning forward')
    out = [0.30, 0.35, TABLE_Z + 0.05]
    check(p.reach_lean(out) is None, 'person: a point on the other side of the table is out of reach')
    # carrying: the item follows the palm
    mug = task.soup
    mug.place([0.30, -0.35, TABLE_Z], 0.0)
    p.set_palm(p.palm_above(mug.top_z(), mug.position()[:2]))
    p.grab(mug)
    p.set_palm(np.array([0.32, -0.30, TABLE_Z + 0.20]))
    expect = p.palm_world() - [0, 0, 0.016 + 0.004 + mug.height]
    check(np.linalg.norm(mug.position() - expect) < 2e-3 and mug.state == 'carried' and not mug.shape.is_dynamic(),
          'person: a carried item follows the palm, kinematic')
    p.let_go()
    p.set_palm(p.palm_world() + [0.0, 0.0, 0.10])           # the hand withdraws
    p._settle_released()
    check(mug.state == 'passive' and mug.shape.is_dynamic(), 'person: a released item is a free body once the hand '
                                                              'has left it')
    mug.hide()
    # an abandoned job: what the hand holds goes back where it was taken from (nothing vanishes). At L1 the person
    # starts with the obstacle in her hand, taken from her side of the table
    set_level('L1')
    np.random.seed(0)
    te.reset()
    task = te._task
    task.dynbench_set_active(True)
    p = task.person
    it = task.obstacle
    home = p.taken_from[it.name][0][:3, 3].copy() if it.name in p.taken_from else None
    q = np.r_[task.robot.arm.get_joint_positions(), 1.0]
    for _ in range(60):                            # she starts putting it in front of the jar ...
        te.step(q)
        if p.job is not None and p.job.name == 'item_in_front_of_jar' and p.held is it and p.job.k >= 1:
            break
    p.fail_job('test')                             # ... and gives up on the way
    for _ in range(300):
        te.step(q)
    check(home is not None and it.state == 'passive' and np.linalg.norm(it.position()[:2] - home[:2]) < 0.01,
          'person: an item held when a job is abandoned is put back where it was taken from')
    set_level('L0')
    np.random.seed(0)
    te.reset()
    task = te._task
    p = task.person
    # flinch: the robot runs into the person's hand while she holds it still in the workspace
    p.reset()
    p.place(-1, 0.30)
    hand = np.array([0.30, -0.25, TABLE_Z + 0.18])
    p.set_palm(hand)
    task.dynbench_set_active(True)
    task.submit(Job('hold_test', [Hold(30.0)]))
    # the gripper comes down onto the hand (its tip 5 cm below the palm: the gripper's body meets the hand)
    path = task.robot.arm.get_path(list(hand - [0.0, 0.0, 0.05]),
                                   quaternion=list(task.robot.arm.get_tip().get_quaternion()), ignore_collisions=True)
    pts = np.asarray(path._path_points).reshape(-1, 7)
    flinched = False
    for qk in pts[::2]:
        te.step(np.r_[qk, 1.0])
        flinched = flinched or any(e['event'] == 'person_flinches' for e in task.event_log)
        if flinched:
            break
    check(flinched, 'person: flinches when the robot runs into her hand')
    check(task.monitor.summary()['collided'], 'monitor: a contact with the person is a collision')
    p.hide()
    check(p.hidden and all(not s.is_respondable() for s in p.parts), 'person: hidden, not respondable')
    set_level(None)


def test_item_contact(env):
    set_level('L0')
    te = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
    np.random.seed(1)
    te.reset()
    task = te._task
    task.monitor.reset()
    it = task.obstacle
    j = task.robot.arm.joints
    link = 0.5 * (np.asarray(j[3].get_position()) + np.asarray(j[4].get_position()))   # on an arm link
    it.place([link[0], link[1], link[2] - 0.5 * it.height], 0.0, passive=False)
    task.monitor.update(0.0, 0.05, task._person_shapes(), task._item_shapes())
    s = task.monitor.summary()
    check(not s['collided'] and s['item_contact_steps'].get(it.name, 0) == 1,
          'monitor: touching an item is counted, not a collision')
    set_level(None)


def test_activation_and_demo(env):
    set_level('L1')
    te = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
    np.random.seed(1)
    te.reset()
    task = te._task
    q = np.r_[task.robot.arm.get_joint_positions(), 1.0]
    palm0 = task.person.palm_world()
    seat0 = (task.person.side, round(task.person.seat_x, 6))
    task.dynbench_set_active(False)
    for _ in range(40):
        te.step(q)
    check(np.allclose(task.person.palm_world(), palm0), 'inactive: the person does not move')
    task.dynbench_set_active(True)
    for _ in range(40):
        te.step(q)
    check(not np.allclose(task.person.palm_world(), palm0), 'active: the person acts (brings the obstacle right after the start)')
    task._external_activation = False
    np.random.seed(1)
    demo = record_static_demo(te)
    check(len(demo) > 20, 'demo recorded with the dynamic task loaded')
    check(task.person.hidden, 'demo mode: the person is out of the scene while RLBench records the demo')
    te.reset_to_demo(demo)
    check(not task.person.hidden and (task.person.side, round(task.person.seat_x, 6)) == seat0,
          'reset_to_demo: the person is back, seated as for the same seed')
    set_level('off')
    te.reset()
    check(task.person.hidden and not task.dynbench_object_handles(), "level 'off': the static task, no person")
    set_level(None)
    os.environ['DYNBENCH_LEVEL'] = 'L2'
    check(core.current_level() == 'L2', 'DYNBENCH_LEVEL is read')
    os.environ.pop('DYNBENCH_LEVEL')
    check(core.current_level() == core.DEFAULT_LEVEL, 'default level')


def test_keyframes(env):
    from rlbench.dynamic_bench.keyframes import static_episode
    set_level('L0')                  # the obstacle stands in front of the jar: the blind replay gets up to it
    te = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
    _, _, demo, kfs = static_episode(te, 2)
    task = te._task
    opens = [k['gripper_open'] for k in kfs]
    check(len(kfs) >= 4 and 0.0 in opens and opens[-1] == 1.0,
          f'static keyframes: {len(kfs)} from the static demo, with the grasp and the release')
    check(task.person.hidden is False and task.keyframes == kfs, 'static keyframes: the episode is reset to the demo '
                                                                 'with the person, the task knows the keyframes')
    arm = task.robot.arm
    for obs in demo:
        te.step(np.r_[obs.joint_positions, 1.0 if obs.gripper_open > 0.5 else 0.0])
    m = task.dynbench_metrics()
    check(m['keyframes']['reached'] >= 2, f'static keyframes: tracked along the blind replay '
                                          f'({m["keyframes"]["reached"]}/{m["keyframes"]["n"]})')
    set_level(None)


def test_human_speed(env):
    """The person's speeds scale with the human speed scale (ablations); the metrics report the actual speeds."""
    from rlbench.dynamic_bench import set_human_speed_scale
    set_level('L1')
    te = env.get_task(get_task_class('dyn_slide_block_target_taken'))
    out = {}
    for scale in (1.0, 0.5):
        set_human_speed_scale(scale)
        np.random.seed(0)
        te.reset()
        task = te._task
        task.dynbench_set_active(True)
        q = np.r_[task.robot.arm.get_joint_positions(), 1.0]
        for _ in range(240):
            te.step(q)
        m = task.dynbench_metrics()
        out[scale] = (m.get('t_obstacle_placed'), m['speeds'])
    (t1, s1), (t2, s2) = out[1.0], out[0.5]
    check(s1['human_speed_scale'] == 1.0 and s2['human_speed_scale'] == 0.5 and t1 is not None and t2 is not None
          and 1.6 < t2 / t1 < 2.4 and 0.4 < s2['person_hand_peak'] / s1['person_hand_peak'] < 0.6,
          f'human speed scale: half speed puts the obstacle down {t2 / t1 if t1 and t2 else 0:.2f}x later, hand peak '
          f'{s1["person_hand_peak"]:.2f} -> {s2["person_hand_peak"]:.2f} m/s')
    check(s1['robot_tip_peak'] < 0.01 and s1['ratio_peak'] is None,
          'speed metrics: a still robot has no person / robot speed ratio')
    set_human_speed_scale(None)
    set_level(None)


def main():
    env = make_env()
    try:
        test_items(env)
        test_person(env)
        test_item_contact(env)
        test_activation_and_demo(env)
        test_keyframes(env)
        test_human_speed(env)
    finally:
        env.shutdown()
    print(f'\n{len(FAILS)} failure(s)' + (': ' + '; '.join(FAILS) if FAILS else ''))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
