"""
A scripted robot with privileged knowledge of the scene, used to check that the benchmark scenarios play out as
designed and can be solved (a test harness, not a baseline). It follows the STATIC keyframes of the episode
(keyframes.py: what a keyframe policy decided on the static scene) and, between them, does what the layer under test
should do about the obstacles: re-plan a segment around an obstacle, wait for the person, park what it holds, move or
push an obstacle away, then go on with the same keyframes.

Motions: RLBench's planners (OMPL for free moves, collision-aware with the scene as it is when the move starts; linear
paths for the keyframes the demo reached ignoring collisions), executed through the task environment (absolute joint
position targets), so the benchmark's dynamics run at every simulation step.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np

from rlbench.dynamic_bench.assets import matrix_to_quat, quat_to_matrix
from rlbench.dynamic_bench.core import TABLE_Z


class Timeout(Exception):
    pass


def _rz(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class ScriptedRobot:

    def __init__(self, task_env, max_steps: int = 6000, log: Optional[Callable[[str], None]] = None):
        self.te = task_env
        self.task = task_env._task
        self.arm = self.task.robot.arm
        self.gripper = self.task.robot.gripper
        self.max_steps = max_steps
        self.steps = 0
        self.open = 1.0
        self.log = log or (lambda s: None)
        self.success = False
        # the Panda fingers close along the tip's y axis; the start pose gives the preferred wrist turn
        R0 = quat_to_matrix(self.arm.get_tip().get_quaternion())
        x0 = R0[:3, 0].copy()
        x0[2] = 0.0
        self.x0 = x0 / np.linalg.norm(x0)

    # --- low level --------------------------------------------------------------------------------------------------

    def act(self, q: Sequence[float]) -> None:
        if self.steps >= self.max_steps:
            raise Timeout(f'{self.steps} steps')
        self.te.step(np.r_[np.asarray(q, dtype=np.float64), self.open])
        self.steps += 1
        ok, _ = self.task.success()
        self.success = self.success or ok

    def q(self) -> np.ndarray:
        return np.asarray(self.arm.get_joint_positions(), dtype=np.float64)

    def hold(self, seconds: float = 0.0, until: Optional[Callable[[], bool]] = None, max_seconds: float = 30.0):
        q = self.q()
        n = int(round((seconds if until is None else max_seconds) / 0.05))
        for _ in range(max(1, n)):
            if until is not None and until():
                return True
            self.act(q)
        return until is None

    def safe_pose(self) -> np.ndarray:
        """Where the robot waits: gripper up and back towards its base, away from the person's side."""
        return np.array([0.05, -0.12 * self.task.person.side, TABLE_Z + 0.45])

    def go_safe(self) -> None:
        q = matrix_to_quat(quat_to_matrix(self.arm.get_tip().get_quaternion()))
        self.move(self.safe_pose(), q)

    def wait_person(self, max_seconds: float = 40.0) -> None:
        """Waits while the person is doing something (a polite robot), out of the way: gripper up and back towards
        the robot's base, away from the side the person sits on (what it holds, it keeps)."""
        p = self.task.person
        if p.job is None and not p.queue:
            return
        tip = self.tip()
        q = matrix_to_quat(quat_to_matrix(self.arm.get_tip().get_quaternion()))
        safe = self.safe_pose()
        if np.linalg.norm(tip - safe) > 0.08:
            self.move(tip + [0, 0, max(0.0, safe[2] - tip[2])], q, linear=True, ignore_collisions=True) or None
            self.move(safe, q)
        self.hold(until=lambda: p.job is None and not p.queue, max_seconds=max_seconds)

    def set_gripper(self, open_: bool) -> None:
        self.open = 1.0 if open_ else 0.0
        q = self.q()
        for _ in range(3):
            self.act(q)

    def top_down(self, finger_yaw: float, flip: bool = False) -> np.ndarray:
        """Quaternion of a vertical, top-down gripper whose fingers close along the horizontal direction finger_yaw
        (of the two symmetric solutions, the one closest to the start pose, the wrist joint has limits; flip: the
        other one)."""
        sols = []
        for a in (finger_yaw, finger_yaw + np.pi):
            y = np.array([np.cos(a), np.sin(a), 0.0])
            z = np.array([0.0, 0.0, -1.0])
            x = np.cross(y, z)
            sols.append((np.dot(x, self.x0), np.c_[x, y, z]))
        sols.sort(key=lambda s: -s[0])
        return matrix_to_quat(sols[1 if flip else 0][1])

    polite = True

    def person_close(self, dist: float = 0.22) -> bool:
        p = self.task.person
        if not self.polite or p.hidden or p.job is None:
            return False
        e, h = p.to_world(p.elbow), p.palm_world()
        tip = self.tip()
        s = np.clip(np.dot(tip - e, h - e) / max(np.dot(h - e, h - e), 1e-9), 0.0, 1.0)
        return bool(np.linalg.norm(tip - (e + s * (h - e))) < dist)

    def follow(self, path, tol: float = 0.02, settle: int = 5) -> None:
        pts = np.asarray(path._path_points).reshape(-1, len(self.arm.joints))
        for k, qk in enumerate(pts[1::2].tolist() + [pts[-1].tolist()]):
            if self.person_close():
                # a polite robot stops while the person's arm is near and goes on when it has left (or after 15 s)
                self.hold(until=lambda: not self.person_close(0.28), max_seconds=15.0)
            for _ in range(settle):
                self.act(qk)
                if np.max(np.abs(self.q() - qk)) < tol:
                    break
        for _ in range(10):
            if np.max(np.abs(self.q() - pts[-1])) < 0.01:
                break
            self.act(pts[-1])

    def move(self, pos, quat, linear: bool = False, ignore_collisions: bool = False, steps: int = 40) -> bool:
        path, err = None, None
        for attempt in range(1 if linear else 3):         # OMPL is randomised: a free move is tried again
            try:
                if linear:
                    path = self.arm.get_linear_path(list(pos), quaternion=list(quat), steps=steps,
                                                    ignore_collisions=ignore_collisions)
                else:
                    path = self.arm.get_path(list(pos), quaternion=list(quat), ignore_collisions=ignore_collisions,
                                             trials=200, max_configs=6, trials_per_goal=6)
                break
            except Exception as e:
                err = e
        if path is None:
            self.log(f'  no path to {np.round(pos, 3)} ({"linear" if linear else "free"}): {type(err).__name__}')
            return False
        self.follow(path)
        return True

    def tip(self) -> np.ndarray:
        return np.asarray(self.arm.get_tip().get_position(), dtype=np.float64)

    # --- skills -----------------------------------------------------------------------------------------------------

    def pick(self, obj_shape, grasp_point, finger_yaw: float, approach: float = 0.12, lift: float = 0.15,
             quat=None, alt_yaws=()) -> bool:
        """Top-down grasp with the fingers along finger_yaw (or the orientation quat), tip at grasp_point; lifts.
        alt_yaws: other finger directions to try when the first one cannot be planned (a cube: +90 degrees)."""
        g = np.asarray(grasp_point, dtype=np.float64)
        self.set_gripper(True)
        options = [quat] if quat is not None else [self.top_down(a, f) for a in (finger_yaw,) + tuple(alt_yaws)
                                                   for f in (False, True)]
        # far objects: a lower approach point may be reachable when the usual one is not; the straight descent from
        # the approach point must be feasible too (else the next option)
        for q, a in [(q, a) for a in (approach, 0.06) for q in options]:
            if self.move(g + [0, 0, a], q) and self.move(g, q, linear=True, ignore_collisions=True):
                approach = a
                break
        else:
            return False
        self.set_gripper(False)
        if not self.task.is_grasped(obj_shape):
            self.log(f'  grasp of {obj_shape.get_name()} failed')
            self.set_gripper(True)
            self.move(g + [0, 0, approach], q, linear=True, ignore_collisions=True)
            return False
        self.move(g + [0, 0, lift], q, linear=True, ignore_collisions=True)
        return True

    def place(self, release_point, finger_yaw: float, approach: float = 0.12, quat=None) -> bool:
        """Lowers the held object so that the tip reaches release_point, opens, retreats up."""
        q = self.top_down(finger_yaw) if quat is None else quat
        r = np.asarray(release_point, dtype=np.float64)
        if not self.move(r + [0, 0, approach], q):
            return False
        self.move(r, q, linear=True, ignore_collisions=True)
        self.set_gripper(True)
        self.move(r + [0, 0, approach], q, linear=True, ignore_collisions=True)
        return True

    def push(self, start, direction, distance: float, height: float, steps: int = 60) -> bool:
        """Closed gripper down at start (tip at height above the table), pushes along direction, lifts."""
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        a = float(np.arctan2(d[1], d[0]))
        s = np.array([start[0], start[1], TABLE_Z + height])
        self.set_gripper(False)
        # fingers across the push (a flat pusher), else along it; either wrist turn; the straight descent from the
        # approach point must be feasible too
        for q in [self.top_down(a + np.pi / 2), self.top_down(a + np.pi / 2, True), self.top_down(a),
                  self.top_down(a, True)]:
            if self.move(s + [0, 0, 0.12], q):
                if self.move(s, q, linear=True, ignore_collisions=True):
                    break
                self.move(s + [0, 0, 0.12], q, linear=True, ignore_collisions=True)
        else:
            self.retreat(0.10)
            return False
        if not self.move(s + d * distance, q, linear=True, ignore_collisions=True, steps=steps):
            self.retreat(0.12)                     # the push itself cannot be done this way
            return False
        self.move(s + d * distance + [0, 0, 0.12], q, linear=True, ignore_collisions=True)
        self.set_gripper(True)
        return True

    # --- static keyframes -------------------------------------------------------------------------------------------

    def to_keyframe(self, kf: dict) -> bool:
        """Tip to the keyframe pose (collision-aware path planned now, or a linear path for a keyframe the demo
        reached ignoring collisions), then the gripper as the keyframe says."""
        pos, quat = np.asarray(kf['position'], dtype=np.float64), kf['quaternion']
        if kf.get('ignore_collisions', 0.0) > 0.5:
            ok = self.move(pos, quat, linear=True, ignore_collisions=True) or \
                self.move(pos, quat, ignore_collisions=True)
        else:
            ok = self.move(pos, quat)
        if not ok and np.linalg.norm(self.tip() - pos) > 0.01:
            return False
        # (a keyframe that only turns the wrist where the tip already is still opens / closes the gripper)
        want_open = kf.get('gripper_open', 1.0) > 0.5
        if want_open != (self.open > 0.5):
            self.set_gripper(want_open)
        return True

    def make_room(self, dist: float = 0.30, always: bool = False) -> None:
        """Out of the person's way (up and back towards the base, away from her side) before waiting: always, or if
        her arm is working within dist of the gripper."""
        if (always or self.person_close(dist)) and np.linalg.norm(self.tip() - self.safe_pose()) > 0.10:
            self.log('  out of the person\'s way')
            self.go_safe()

    def person_near_way(self, target, radius: float = 0.25) -> bool:
        """The person is doing something with her hand near the way from the tip to target (horizontally)."""
        p = self.task.person
        if p.hidden or (p.job is None and not p.queue):
            return False
        if p.palm_world()[2] < TABLE_Z:                  # the hands are back in the lap
            return False
        a, b = self.tip()[:2], np.asarray(target, dtype=np.float64)[:2]
        pts = [p.palm_world()[:2], p.to_world(p.elbow)[:2]] + ([p.held.position()[:2]] if p.held is not None else [])
        for q in pts:
            u = np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.0, 1.0)
            if np.linalg.norm(q - (a + u * (b - a))) < radius:
                return True
        return False

    def follow_keyframes(self, kfs, start: int = 0, stop: Optional[int] = None,
                         before: Optional[Callable[[int], bool]] = None) -> bool:
        """Executes keyframes start..stop-1; before(i) runs before keyframe i (a remedy; False: give up). The robot
        waits while the person's hand works near its next way."""
        stop = len(kfs) if stop is None else stop
        for i in range(start, stop):
            if before is not None and before(i) is False:
                return False
            if self.person_near_way(kfs[i]['position']):
                self.log(f'  the person works near the way to keyframe {i}: wait')
                self.make_room(always=getattr(self, 'room_always', False))
                self.hold(until=lambda: not self.person_near_way(kfs[i]['position']), max_seconds=20.0)
            if not self.to_keyframe(kfs[i]):
                self.log(f'  keyframe {i} cannot be reached as decided')
                return False
        return True

    def pick_along(self, obj_shape, grasp_point, quat, approach_vec, lift: float = 0.05) -> bool:
        """Grasp with a given orientation, coming in along approach_vec (from grasp_point + approach_vec), as a
        static grasp keyframe was reached (a side grasp comes in from the side); lifts."""
        g = np.asarray(grasp_point, dtype=np.float64)
        self.set_gripper(True)
        if not self.move(g + approach_vec, quat):
            return False
        if not self.move(g, quat, linear=True, ignore_collisions=True):
            return False
        self.set_gripper(False)
        if not self.task.is_grasped(obj_shape):
            self.log(f'  grasp of {obj_shape.get_name()} failed')
            self.set_gripper(True)
            self.move(g + approach_vec, quat, linear=True, ignore_collisions=True)
            return False
        self.move(g + [0, 0, lift], quat, linear=True, ignore_collisions=True)
        return True

    def retreat(self, up: float = 0.15) -> None:
        tip = self.tip()
        q = matrix_to_quat(quat_to_matrix(self.arm.get_tip().get_quaternion()))
        self.move(tip + [0, 0, up], q, linear=True, ignore_collisions=True)


# ---------------------------------------------------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------------------------------------------------

def shape_yaw(shape) -> float:
    m = shape.get_matrix()
    return float(np.arctan2(m[1, 0], m[0, 0]))


def item_narrow_yaw(item) -> float:
    """Horizontal direction of the item's narrow side (the fingers close along it)."""
    m = item.frame()
    axis = 0 if item.half_extents[0] <= item.half_extents[1] else 1
    v = m[:3, axis]
    return float(np.arctan2(v[1], v[0]))


def free_table_spot(task, radius: float, avoid: Sequence[Sequence[float]], rng: np.random.Generator,
                    region=((0.10, 0.45), (-0.32, 0.32)), clearance: float = 0.09,
                    prefer: Optional[Callable[[np.ndarray], bool]] = None, tries: int = 300) -> Optional[np.ndarray]:
    """A table point at least clearance + radius from every object of the scene and from the avoid points."""
    from pyrep.const import ObjectType
    pts = [s.get_position()[:2] for s in task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE)
           if s.is_respondable() and s.get_position()[2] < TABLE_Z + 0.4]
    pts += [it.position()[:2] for it in task.items.values() if not it.hidden]
    pts += [np.asarray(a)[:2] for a in avoid]
    pts = np.asarray(pts)
    fallback = None
    for _ in range(tries):
        xy = rng.uniform([region[0][0], region[1][0]], [region[0][1], region[1][1]])
        if len(pts) and np.min(np.linalg.norm(pts - xy, axis=1)) < clearance + radius:
            continue
        if prefer is None or prefer(xy):
            return np.array([xy[0], xy[1], TABLE_Z])
        fallback = fallback if fallback is not None else np.array([xy[0], xy[1], TABLE_Z])
    return fallback
