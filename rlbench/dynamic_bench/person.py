"""
The person of the benchmark: an adult seated on a dining chair at the long side of the table (CoppeliaSim's 'Sitting
Bill' meshes, scaled by 1.2), who reaches into the workspace with the right arm.

Physics. The person is an AGENT: kinematic (the robot cannot push it), moved by the task script. Every body part has a
simple collision shape (respondable, collidable, not rendered) with the realistic mesh as its child, the convention of
RLBench's task objects (perception labels a mesh by its respondable parent). The parts push the free objects they run
into, so the person can knock things over. Items in the hand are 'carried': kinematic, following the palm; when the
person lets go they become free bodies again.

Arm. Shoulder - elbow - palm chain solved in closed form (two links, elbow down and out), plus a forward lean of the
torso about the hip when the target is far (people lean to reach across a table). The palm moves along minimum-jerk
point-to-point segments (human reaching), lifted to a cruise height between distant points.

Jobs. The person does one job at a time: a list of steps (move the palm, grab, let go, hold, wait for a condition).
Everything the person handles is in the scene and visible: it is taken from where it stands (the person's own things
stand on the table in front of her, out of the robot's way) and, when a job is abandoned with something in the hand,
put back where it was taken from. A job that cannot proceed (target occupied, robot in the way) waits, then gives up
after a timeout, and the task script is told (job.status). At rest the hands are in the lap.

Safety. If the robot touches the person, the person flinches: pulls the hand back to the lap and resumes the job
after a pause (up to MAX_ATTEMPTS times), from the first step after its last grab / release (what was done stays
done). Before moving the palm to a point the robot's gripper is near, the person waits for the robot to leave.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from pyrep.backend import sim
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

from rlbench.dynamic_bench import motion
from rlbench.dynamic_bench.assets import matrix_to_quat, quat_to_matrix, remove_named, remove_tree

TABLE_Z = 0.752
TABLE_HALF_Y = 0.55                  # the dining table spans y in [-0.55, 0.55] and x in [-0.5, 1.1]
TABLE_X = (-0.5, 1.1)
SCALE = 1.2
HIP_FROM_EDGE = 0.20                 # [m] horizontal distance of the hip behind the table edge
MAX_LEAN = np.radians(35.0)
REACH_MARGIN = 0.03                  # [m] the palm is never driven closer than this to full arm extension
HAND_HALF_THICK = 0.016              # [m] palm surface below the palm point (where carried items touch the hand)
CRUISE = 0.22                        # [m] default height of the palm above the table between distant points
ROBOT_KEEP_OUT = 0.15                # [m] the person does not move the palm towards a point this close to a MOVING gripper
KEEP_OUT_STILL = 0.07                # [m] ... or this close to a gripper standing still (she works next to a still robot)
YIELD_DIST = 0.09                    # [m] the palm (or what it carries) stops while a moving gripper is this close
YIELD_STILL = 0.04                   # [m] ... or a still one
YIELD_RESUME = 0.14                  # [m] ... and moves on once the gripper is this far again
STILL_SPEED = 0.02                   # [m/s] the gripper counts as still below this speed
REACTION = 0.3                       # [s] the person notices the gripper this late
PINCH_DROP = 0.012                   # [m] an item pinched by its edge hangs this much below the palm point
FINGERS = 0.05                       # [m] the (curled) finger tips are this far ahead of the palm point
RELEASE_CLEARANCE = 0.01             # [m] a released item becomes a free body once the hand is this far from it
FLINCH_PAUSE = 2.0                   # [s] pause after a contact with the robot before resuming
MAX_ATTEMPTS = 3
ROBOT_WAIT_MAX = 10.0             # [s] longest wait for the robot to leave a point before giving up
NAME = 'dynbench_person'

_COPPELIA = os.environ.get('COPPELIASIM_ROOT', '')
BILL_TTM = os.path.join(_COPPELIA, 'models', 'people', 'Sitting Bill.ttm')
CHAIR_TTM = os.path.join(_COPPELIA, 'models', 'furniture', 'chairs', 'dining chair.ttm')


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rot_x(a: float) -> np.ndarray:
    m = np.eye(4)
    m[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def _lift(dz: float) -> np.ndarray:
    m = np.zeros((4, 4))
    m[2, 3] = dz
    return m


def _frame(origin, z_axis, x_hint) -> np.ndarray:
    """4x4 frame with z along z_axis and x as close as possible to x_hint."""
    z = _unit(np.asarray(z_axis, dtype=np.float64))
    x = np.asarray(x_hint, dtype=np.float64) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
    x = _unit(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, origin
    return m


# -------------------------------------------------------------------------------------------------------------------
# Steps of a job
# -------------------------------------------------------------------------------------------------------------------

class Step:
    irreversible = False             # grab / let go / appear / vanish: a resumed job does not redo them

    def start(self, person: 'Person', t: float) -> None:
        pass

    def run(self, person: 'Person', t: float) -> bool:
        """Advances the step; True when it is finished."""
        return True


class Move(Step):
    """Palm to a point along minimum-jerk segments, through a cruise height between distant points.
    free_check: the point must not be blocked (an item to put down there would fit) before the palm descends."""

    def __init__(self, target, speed: float = 0.45, cruise: Optional[float] = None,
                 direct: bool = False, wait_for_robot: bool = True):
        """:param target: a point, or a function returning it (evaluated when the step starts)."""
        self.target_fn = target if callable(target) else (lambda: target)
        self.speed, self.cruise, self.direct, self.wait_for_robot = float(speed), cruise, direct, wait_for_robot

    def start(self, person, t):
        self.target = np.asarray(self.target_fn(), dtype=np.float64)
        if person.reach_lean(self.target) is None:
            # out of reach: the job fails rather than put something down in the wrong place
            person.fail_job('out of reach')
            self.segments = []
            return
        self.segments = person.path_segments(person.palm_target, self.target, self.speed, self.cruise, self.direct,
                                             check_arm=True)
        self.k, self.t0, self.started, self.waiting_since = 0, t, False, None

    def run(self, person, t):
        while self.k < len(self.segments):
            seg = self.segments[self.k]
            if not self.started:
                # a person does not move the hand to where the robot's gripper is: wait before such a segment
                if self.wait_for_robot and (person.robot_near(seg.p1) or not person.path_clear(seg)):
                    if self.waiting_since is None:
                        self.waiting_since = t
                        person.task.log_event('person_waits_for_robot', at=[round(float(x), 3) for x in seg.p1])
                    elif t - self.waiting_since > ROBOT_WAIT_MAX:
                        person.fail_job('robot in the way')
                    return False
                self.started, self.t0, self.waiting_since, self.paused = True, t, None, False
            # ... and stops when the gripper comes too close to the hand (after a reaction time)
            still = person.robot_still()
            near = YIELD_STILL if still else (YIELD_RESUME if self.paused else YIELD_DIST)
            if self.wait_for_robot and person.gripper_close(near):
                person.close_for = getattr(person, 'close_for', 0.0) + person.dt
                # a still robot is seen in advance; a moving one is noticed after the reaction time
                if self.paused or person.close_for >= REACTION or still:
                    if not self.paused:
                        self.paused, self.pause_start = True, t
                        person.task.log_event('person_pauses')
                    elif t - self.pause_start > ROBOT_WAIT_MAX:
                        person.fail_job('robot in the way')
                        return False
                    self.t0 += person.dt
                    return False
            elif self.wait_for_robot and person.arm_hits_robot(seg.at(t - self.t0 + 0.25)):
                # the robot (its arm, not only the gripper) stands where the arm is going: wait for it to leave
                if not self.paused:
                    self.paused, self.pause_start = True, t
                    person.task.log_event('person_pauses')
                elif t - self.pause_start > ROBOT_WAIT_MAX:
                    person.fail_job('robot in the way')
                    return False
                self.t0 += person.dt
                return False
            else:
                person.close_for = 0.0
                self.paused = False
            tau = t - self.t0
            person.set_palm(seg.at(tau))
            if not seg.done(tau):
                return False
            self.k += 1
            self.started = False
        return True


class Hold(Step):
    """Keeps the palm still (optionally swaying, as a hand held out in the air) for a duration or until a condition."""

    def __init__(self, duration: float = np.inf, until: Optional[Callable[[], bool]] = None, sway: float = 0.0,
                 sway_period: float = 2.5):
        self.duration, self.until, self.sway, self.period = float(duration), until, float(sway), float(sway_period)

    def start(self, person, t):
        self.t0, self.p0 = t, person.palm_target.copy()

    def run(self, person, t):
        if self.sway > 0.0:
            a = 2 * np.pi * (t - self.t0) * person.speed_scale / self.period     # a faster person fidgets faster
            person.set_palm(self.p0 + self.sway * np.array([np.sin(a), 0.5 * np.sin(2 * a), 0.3 * np.sin(a)]))
        if self.until is not None and self.until():
            return True
        return t - self.t0 >= self.duration


class WaitUntil(Step):
    """Waits (palm still) until a condition holds; on timeout the job fails (status 'gave_up'), or goes on if
    fail=False."""

    def __init__(self, condition: Callable[[], bool], timeout: float, reason: str, fail: bool = True):
        self.condition, self.timeout, self.reason, self.fail = condition, float(timeout), reason, fail

    def start(self, person, t):
        self.t0 = t

    def run(self, person, t):
        if self.condition():
            return True
        if t - self.t0 > self.timeout:
            if not self.fail:
                return True
            person.fail_job(self.reason)
        return False


class Grab(Step):
    irreversible = True

    def __init__(self, item, grip: str = 'top'):
        self.item, self.grip = item, grip

    def run(self, person, t):
        person.grab(self.item, self.grip)
        return True


class LetGo(Step):
    """The item in the hand becomes a free body (it settles where it is)."""
    irreversible = True

    def run(self, person, t):
        person.let_go()
        return True


class Call(Step):
    irreversible = True

    def __init__(self, fn: Callable[[float], None]):
        self.fn = fn

    def run(self, person, t):
        self.fn(t)
        return True


class Job:
    def __init__(self, name: str, steps: List[Step]):
        self.name, self.steps = name, steps
        self.k = 0
        self.status = 'pending'          # pending, running, done, gave_up
        self.attempts = 0
        self.reason = ''
        self.preemptible = False         # a scene change: the task's jobs may interrupt it


# -------------------------------------------------------------------------------------------------------------------
# The person
# -------------------------------------------------------------------------------------------------------------------

class Person:

    def __init__(self, task):
        self.task = task
        remove_named(NAME)
        self.root = Dummy.create(size=0.05)
        self.root.set_name(NAME)
        self._build()
        self.job: Optional[Job] = None
        self.queue: List[Job] = []
        self.held = None
        self.releasing: List = []
        self.taken_from: Dict[str, tuple] = {}     # item name -> (frame, grip) where the person took it from
        self.speed_scale = 1.0                     # every speed of the person is multiplied by this (ablations)
        self.hidden = False
        self.side, self.seat_x = -1, 0.3
        self.place(-1, 0.3)
        self.hide()

    # --- construction -----------------------------------------------------------------------------------------------

    def _build(self) -> None:
        pr = self.task.pyrep
        bill = pr.import_model(BILL_TTM)
        tree = [bill] + bill.get_objects_in_tree(exclude_base=True)
        sim.simScaleObjects([o.get_handle() for o in tree], SCALE, True)
        objs = {o.get_name().split('#')[0]: o for o in tree}
        hip_w = np.asarray(objs['Bill_legJoint'].get_position(), dtype=np.float64)
        # person frame P (rest pose): origin at the hip, x = the person's right, y = forward, z = up
        # (Sitting Bill faces -y: forward = -y, right = -x)
        w_p = np.eye(4)
        w_p[:3, 0], w_p[:3, 1], w_p[:3, 2], w_p[:3, 3] = [-1, 0, 0], [0, -1, 0], [0, 0, 1], hip_w
        inv = np.linalg.inv(w_p)

        def p_of(name):
            return (inv @ np.r_[objs[name].get_position(), 1.0])[:3]

        def mesh_p(shape):
            v, _, _ = shape.get_mesh_data()
            v = np.c_[np.asarray(v, dtype=np.float64).reshape(-1, 3), np.ones(len(v))]
            return (v @ (inv @ shape.get_matrix()).T)[:, :3]

        self.hip_height = float(hip_w[2])
        self.S_rest = p_of('Bill_rightShoulderJoint')
        self.E_rest = p_of('Bill_rightElbowJoint')
        fore = mesh_p(objs['Bill_rightForearm'])
        tip = fore[np.argmax(np.linalg.norm(fore - self.E_rest, axis=1))]
        self.L1 = float(np.linalg.norm(self.E_rest - self.S_rest))
        self.L_tip = float(np.linalg.norm(tip - self.E_rest))
        self.L2 = self.L_tip - 0.085 * SCALE            # palm point: 8.5 cm (scaled) back from the finger tips
        self.L_wrist = self.L_tip - 0.18 * SCALE
        self.palm_rest = self.E_rest + _unit(tip - self.E_rest) * self.L2
        self.n_rest = _unit(np.cross(self.E_rest - self.S_rest, self.palm_rest - self.E_rest))
        if np.linalg.norm(self.n_rest) < 1e-6:
            self.n_rest = np.array([1.0, 0.0, 0.0])
        self.n = self.n_rest.copy()

        def proxy(kind, size, matrix_p, name, parent):
            shape = Shape.create(type=kind, size=[float(s) for s in size], mass=5.0, respondable=True, static=True,
                                 renderable=False)
            shape.set_matrix(w_p @ matrix_p)
            shape.set_name(f'{NAME}_{name}_human')
            for prop in (shape.set_collidable, shape.set_measurable, shape.set_detectable):
                prop(True)
            shape.set_parent(parent, keep_in_place=True)
            return shape

        def box_around(shape, shrink=0.9, z_min=None):
            v = mesh_p(shape)
            if z_min is not None:
                v = v[v[:, 2] > z_min]
            lo, hi = v.min(0), v.max(0)
            m = np.eye(4)
            m[:3, 3] = 0.5 * (lo + hi)
            return (hi - lo) * shrink, m

        self.root.set_matrix(w_p)
        # torso (leans about the hip), with the head and the resting left arm
        # the torso: the part of the body mesh above the pelvis (the seated mesh also covers the lap)
        size, m = box_around(objs['Bill_body'], 0.85, z_min=0.10)
        self.torso = proxy(PrimitiveShape.CUBOID, size, m, 'torso', self.root)
        self.torso_rest = inv @ self.torso.get_matrix()
        size, m = box_around(objs['Bill_head'], 0.9)
        self.head = proxy(PrimitiveShape.SPHERE, [size.max()] * 3, m, 'head', self.torso)
        size, m = box_around(objs['Bill_leftUpperArm'], 0.85)
        self.left_arm = proxy(PrimitiveShape.CUBOID, size, m, 'left_upper_arm', self.torso)
        size, m = box_around(objs['Bill_upperLegs'], 0.9)
        self.legs = proxy(PrimitiveShape.CUBOID, size, m, 'legs', self.root)
        # right arm: upper arm (shoulder-elbow), forearm (elbow-wrist), hand (wrist-finger tips), in segment frames
        self.upper = proxy(PrimitiveShape.CYLINDER, [0.10 * SCALE, 0.10 * SCALE, self.L1],
                           self._upper_frame(self.S_rest, self.E_rest, self.n_rest), 'upper_arm', self.root)
        self.forearm = proxy(PrimitiveShape.CYLINDER, [0.07, 0.07, self.L_wrist],                # 7 cm across
                             self._fore_frame(self.E_rest, self.palm_rest, self.n_rest), 'forearm', self.root)
        # the hand's collision shape: from the wrist to the curled finger tips (a hand holding something from above
        # does not stick its fingers out; the mesh shows an open hand)
        self.hand_len = (self.L2 + FINGERS) - self.L_wrist
        self.hand = proxy(PrimitiveShape.CUBOID, [0.085 * SCALE, 2 * HAND_HALF_THICK, self.hand_len],
                          self._hand_frame(self.E_rest, self.palm_rest, self.n_rest), 'hand', self.root)
        self.upper_rest = self._upper_frame(self.S_rest, self.E_rest, self.n_rest)
        self.fore_rest = self._fore_frame(self.E_rest, self.palm_rest, self.n_rest)
        self.hand_rest = self._hand_frame(self.E_rest, self.palm_rest, self.n_rest)
        # the realistic meshes become children of the collision shapes
        attach = {'Bill_body': self.torso, 'Bill_head': self.head, 'Bill_leftUpperArm': self.left_arm,
                  'Bill_leftForearm': self.torso, 'Bill_upperLegs': self.legs, 'Bill_lowerLegs': self.legs,
                  'Bill_shoes': self.legs, 'Bill_rightUpperArm': self.upper, 'Bill_rightForearm': self.forearm}
        self.meshes = []
        for name, parent in attach.items():
            mesh = Shape(objs[name].get_handle())
            mesh.set_parent(parent, keep_in_place=True)
            mesh.set_name(f'{NAME}_{name[5:]}_human_mesh')
            mesh.set_respondable(False)
            mesh.set_dynamic(False)
            self.meshes.append(mesh)
        # what Bill's meshes brought along (his joints and his own collision shapes, e.g. a 10 cm cylinder in each
        # forearm) must go: the person's collision shapes are the proxies above
        keep = {m.get_handle() for m in self.meshes}
        for mesh in self.meshes:
            for o in reversed(mesh.get_objects_in_tree(exclude_base=True)):
                if o.get_handle() not in keep:
                    try:
                        o.remove()
                    except Exception:
                        pass
        # the rest of Bill's model (joints, dummies, his own collision shapes, his script) goes
        for o in reversed(bill.get_objects_in_tree(exclude_base=True)):
            if o.get_handle() not in {m.get_handle() for m in self.meshes}:
                try:
                    o.remove()
                except Exception:
                    pass
        bill.remove()
        # the chair (furniture: static)
        chair = pr.import_model(CHAIR_TTM)
        chair_tree = [chair] + chair.get_objects_in_tree(exclude_base=True)
        sim.simScaleObjects([o.get_handle() for o in chair_tree], SCALE, True)
        for o in chair_tree:
            if o.get_type() == ObjectType.SHAPE:
                s = Shape(o.get_handle())
                s.set_dynamic(False)
        chair.set_model(False)
        self.chair = chair
        self.chair_offset = None
        self.chair.set_parent(self.root, keep_in_place=True)
        self.parts = [self.torso, self.head, self.left_arm, self.legs, self.upper, self.forearm, self.hand]
        self.arm_parts = [self.upper, self.forearm, self.hand]
        # rest palm: on the lap, under the table top
        self.lap = np.array([0.10, 0.10, 0.06])
        self.lean = 0.0
        self.palm_target = self.lap.copy()

    @staticmethod
    def _upper_frame(S, E, n) -> np.ndarray:
        return _frame(0.5 * (S + E), E - S, n)

    def _fore_frame(self, E, H, n) -> np.ndarray:
        d = _unit(H - E)
        return _frame(E + d * 0.5 * self.L_wrist, d, n)

    def _hand_frame(self, E, H, n) -> np.ndarray:
        """The hand is held level (the wrist bends): its long axis is the horizontal direction of the forearm, its
        thickness vertical, from hand_len - FINGERS behind the palm point to FINGERS ahead of it."""
        d = H - E
        d[2] = 0.0
        d = _unit(d) if np.linalg.norm(d) > 1e-6 else np.array([0.0, 1.0, 0.0])
        side = np.cross([0.0, 0.0, 1.0], d)
        return _frame(H + d * (FINGERS - 0.5 * self.hand_len), d, side)

    # --- placement --------------------------------------------------------------------------------------------------

    def world_from_person(self, side: int, seat_x: float) -> np.ndarray:
        """Person frame in the world when seated at the side (-1: y < 0, +1: y > 0) of the table, hip at x = seat_x."""
        fwd = np.array([0.0, -float(side), 0.0])
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = right, fwd, [0.0, 0.0, 1.0]
        m[:3, 3] = [seat_x, side * (TABLE_HALF_Y + HIP_FROM_EDGE), self.hip_height]
        return m

    def place(self, side: int, seat_x: float) -> None:
        self.side, self.seat_x = int(side), float(seat_x)
        self.W = self.world_from_person(self.side, self.seat_x)
        self.W_inv = np.linalg.inv(self.W)
        self.root.set_matrix(self.W)
        self.lap_world = self.to_world(self.lap)
        self.palm_target = self.lap_world.copy()
        self._pose_arm(self.lap)

    def to_world(self, p_person) -> np.ndarray:
        return (self.W @ np.r_[p_person, 1.0])[:3]

    def to_person(self, p_world) -> np.ndarray:
        return (self.W_inv @ np.r_[np.asarray(p_world, dtype=np.float64), 1.0])[:3]

    def shoulder(self, lean: float) -> np.ndarray:
        return (_rot_x(-lean) @ np.r_[self.S_rest, 1.0])[:3]

    def reach_lean(self, p_world) -> Optional[float]:
        """Smallest torso lean that lets the palm reach the point (None: out of reach)."""
        p = self.to_person(p_world)
        lmax = self.L1 + self.L2 - REACH_MARGIN
        for lean in np.linspace(0.0, MAX_LEAN, 15):
            if np.linalg.norm(p - self.shoulder(lean)) <= lmax:
                return float(lean)
        return None

    def can_reach(self, points: Sequence[Sequence[float]], side: int, seat_x: float) -> bool:
        keep = (self.W.copy(), self.W_inv.copy())
        self.W = self.world_from_person(side, seat_x)
        self.W_inv = np.linalg.inv(self.W)
        try:
            return all(self.reach_lean(p) is not None for p in points)
        finally:
            self.W, self.W_inv = keep

    def choose_seat(self, points: Sequence[Sequence[float]], rng: np.random.Generator,
                    sides: Sequence[int] = (-1, 1)) -> bool:
        """Seats the person where every point is within reach: the side closer to the points first (random on a tie),
        hip x near the points; False if no seat works."""
        pts = np.asarray(points, dtype=np.float64)
        mean_y = float(pts[:, 1].mean())
        order = sorted(sides, key=lambda s: (-s * mean_y, rng.random()))
        xs = float(np.clip(pts[:, 0].mean(), -0.1, 0.8))
        for side in order:
            for dx in (0.0, 0.08, -0.08, 0.16, -0.16, 0.24):
                x = float(np.clip(xs + dx, -0.15, 0.85))
                if self.can_reach(pts, side, x):
                    self.place(side, x)
                    return True
        return False

    # --- arm --------------------------------------------------------------------------------------------------------

    def _lean_for(self, p: np.ndarray) -> float:
        """Torso lean for a palm target in the person frame: grows smoothly with how far forward the target is, at
        least what the reach needs."""
        lmax = self.L1 + self.L2 - REACH_MARGIN
        soft = MAX_LEAN * float(np.clip((p[1] - 0.45) / 0.35, 0.0, 1.0))
        lean = soft
        while np.linalg.norm(p - self.shoulder(lean)) > lmax and lean < MAX_LEAN:
            lean = min(MAX_LEAN, lean + np.radians(1.0))
        return lean

    def _pose_arm(self, p: np.ndarray) -> np.ndarray:
        """Poses torso and right arm for a palm target p (person frame); returns the palm actually reached."""
        lean = self._lean_for(p)
        S = self.shoulder(lean)
        lmax = self.L1 + self.L2 - REACH_MARGIN
        d_vec = p - S
        d = float(np.linalg.norm(d_vec))
        if d > lmax:
            p = S + d_vec * (lmax / d)
            d = lmax
        d = max(d, abs(self.L1 - self.L2) + 1e-3)
        u = d_vec / max(np.linalg.norm(d_vec), 1e-9)
        a = (self.L1 ** 2 - self.L2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(self.L1 ** 2 - a ** 2, 0.0))
        pole = _unit(np.array([0.9, -0.1, -0.4]))            # elbow out (to the right), a little down
        v = pole - np.dot(pole, u) * u
        v = _unit(v) if np.linalg.norm(v) > 1e-6 else _unit(np.cross(u, [1.0, 0.0, 0.0]))
        E = S + a * u + h * v
        n = np.cross(E - S, p - E)
        if np.linalg.norm(n) > 1e-6:
            n = _unit(n)
            if np.dot(n, self.n) < 0:
                n = -n
            self.n = n
        self.lean = lean
        self.torso.set_matrix(self.W @ _rot_x(-lean) @ self.torso_rest)
        self.upper.set_matrix(self.W @ self._upper_frame(S, E, self.n))
        self.forearm.set_matrix(self.W @ self._fore_frame(E, p, self.n))
        self.hand.set_matrix(self.W @ self._hand_frame(E, p, self.n))
        self.elbow, self.palm = E, p
        return p

    def set_palm(self, p_world: Sequence[float]) -> None:
        p = self.to_person(p_world)
        self.palm_target = np.asarray(p_world, dtype=np.float64).copy()
        reached = self._pose_arm(p)
        if self.held is not None:
            self._carry(self.to_world(reached))

    def palm_world(self) -> np.ndarray:
        return self.to_world(self.palm)

    # --- items in the hand ------------------------------------------------------------------------------------------

    def grab(self, item, grip: str = 'top') -> None:
        """grip 'top': the palm on the item; 'pinch': the item held by its edge in front of the finger tips."""
        self.taken_from.setdefault(item.name, (item.frame(), grip))
        item.make_carried()
        self.held = item
        self.grip = grip
        self._held_yaw = item.yaw()
        # the item keeps its pose relative to the palm (no hand rotation is modelled: a hand carries things level)
        self._held_rel = item.frame()
        self._held_rel[:3, 3] -= self.palm_world()
        self.task.log_event('person_grabs', object=item.name, grip=grip)

    def let_go(self) -> None:
        """The hand opens: the item stays where it is and becomes a free body as soon as no part of the person
        touches it any more (a wide item may still touch the forearm while the hand withdraws: a free body inside a
        kinematic one would be thrown out by the physics engine)."""
        if self.held is None:
            return
        item = self.held
        self.held = None
        self.releasing.append(item)
        self.task.log_event('person_lets_go', object=item.name,
                            at=[round(float(x), 4) for x in item.position()])
        self._settle_released()

    def _settle_released(self) -> None:
        for item in list(self.releasing):
            if item.state != 'carried':
                self.releasing.remove(item)
                continue
            if self.task.is_grasped(item.shape) or not self._near_body(item, RELEASE_CLEARANCE):
                item.make_passive()
                self.releasing.remove(item)

    def _carry(self, palm_w: np.ndarray) -> None:
        item = self.held
        if self.task.is_grasped(item.shape):          # the robot took it (hand-over): it is no longer the person's
            return
        if getattr(self, 'grip', 'top') == 'pinch':
            fwd = self._pinch_dir(palm_w)
            bottom = palm_w + fwd * self._pinch_reach(item) - np.array([0.0, 0.0, PINCH_DROP + 0.5 * item.height])
            item.set_pose(bottom, [0.0, 0.0, np.sin(0.5 * self._held_yaw), np.cos(0.5 * self._held_yaw)])
            return
        frame = self._held_rel.copy()
        frame[:3, 3] += palm_w
        item.set_frame(frame)

    def _near_body(self, item, gap: float) -> bool:
        """Some part of the person touches the item or is closer than gap to it."""
        h = item.shape.get_handle()
        was = item.shape.is_measurable()
        item.shape.set_measurable(True)
        try:
            for part in self.parts:
                if sim.simCheckCollision(part.get_handle(), h) == 1:
                    return True
                try:
                    d = sim.simCheckDistance(part.get_handle(), h, gap)[6]
                except Exception:
                    d = 0.0
                if 0.0 < d < gap:
                    return True
            return False
        finally:
            item.shape.set_measurable(was)

    def _pinch_dir(self, point_w) -> np.ndarray:
        """Horizontal direction from the shoulder to a point (the hand points that way when pinching)."""
        d = np.asarray(point_w, dtype=np.float64) - self.to_world(self.S_rest)
        d[2] = 0.0
        return d / max(np.linalg.norm(d), 1e-9)

    @staticmethod
    def _pinch_reach(item) -> float:
        """Palm-to-item-centre distance of a pinch: the finger tips 3 mm short of the item's edge (they must not push
        into a free item on the way to grabbing it)."""
        return FINGERS + item.radius + 0.003

    def palm_pinching(self, item, bottom_w) -> np.ndarray:
        """Palm point that holds the item by its edge (in front of the finger tips) with its bottom at bottom_w."""
        b = np.asarray(bottom_w, dtype=np.float64)
        centre = b + [0.0, 0.0, 0.5 * item.height]
        fwd = self._pinch_dir(centre)
        return centre - fwd * self._pinch_reach(item) + [0.0, 0.0, PINCH_DROP]

    @staticmethod
    def palm_above(item_top_z: float, xy: Sequence[float]) -> np.ndarray:
        """Palm point to hold an item from above (the palm surface 4 mm over its top)."""
        return np.array([xy[0], xy[1], item_top_z + HAND_HALF_THICK + 0.004])

    # --- paths ------------------------------------------------------------------------------------------------------

    def path_segments(self, p0, p1, speed, cruise=None, direct=False, check_arm=False) -> List[motion.Segment]:
        """Minimum-jerk segments from p0 to p1: out of / back into the lap vertically over the table edge, through a
        cruise height between points more than 5 cm apart."""
        p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        cz = TABLE_Z + (CRUISE if cruise is None else cruise)
        # over tall things on the way (a rack, a bottle, the cupboard): the hand and the forearm pass 10 cm above them
        top = self.task.highest_along(p0[:2], p1[:2], width=0.12)
        hanging = 0.0                                   # what the hand carries hangs below it
        if self.held is not None:
            hanging = (0.5 * self.held.height + PINCH_DROP if getattr(self, 'grip', 'top') == 'pinch'
                       else self.held.height + 0.004)
        cz = max(cz, top + 0.10 + HAND_HALF_THICK + hanging)
        if check_arm and not direct:
            # the whole arm (the elbow may hang lower than the hand) must clear what is on the table along the way:
            # higher until it does
            for _ in range(6):
                if self._travel_clear(p0, p1, cz):
                    break
                cz += 0.05
        return self._segments(p0, p1, speed * self.speed_scale, cz, direct)

    def _travel_clear(self, p0, p1, cz) -> bool:
        """The arm touches nothing at points of the travel at height cz between p0 and p1 (not at the ends: the
        descent is the job's business)."""
        a = np.array([p0[0], p0[1], cz])
        b = np.array([p1[0], p1[1], cz])
        for s in (0.2, 0.4, 0.6, 0.8):
            if self.reach_lean(a + (b - a) * s) is not None and not self.arm_fits(a + (b - a) * s):
                return False
        return True

    def _segments(self, p0, p1, speed, cz, direct) -> List[motion.Segment]:
        above_lap = self.lap_world.copy()
        above_lap[2] = TABLE_Z + 0.08
        pts = [p0]
        going_down = p1[2] < TABLE_Z
        far = np.linalg.norm((above_lap if going_down else p1)[:2] - p0[:2]) > 0.05
        if p0[2] < TABLE_Z or (not direct and far and p0[2] < cz - 0.03):
            # first straight up to the travel height (out of the lap behind the table edge, or off the table where
            # the hand just worked): the forearm does not sweep the objects near the start
            pts.append(np.array([p0[0], p0[1], cz]))
        goal = above_lap if going_down else p1
        if not direct and np.linalg.norm(goal[:2] - pts[-1][:2]) > 0.05:
            pts.append(np.array([goal[0], goal[1], max(cz, goal[2])]) if not going_down else
                       np.array([goal[0], goal[1], max(pts[-1][2], goal[2])]))
        if going_down:                                       # into the lap: over it at the travel height, then down
            pts[-1] = np.array([above_lap[0], above_lap[1], max(pts[-1][2], cz)])
            pts.append(above_lap)
        pts.append(p1)
        segs = []
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) > 1e-4:
                segs.append(motion.Segment(a, b, speed, 'min_jerk'))
        return segs or [motion.Segment(p0, p1, speed, 'min_jerk')]

    def _keep_out(self) -> float:
        return KEEP_OUT_STILL if self.robot_still() else ROBOT_KEEP_OUT

    def robot_near(self, p_world) -> bool:
        """A moving gripper is within ROBOT_KEEP_OUT of the point horizontally, at a height where it is in the way; a
        still one within KEEP_OUT_STILL of it."""
        tip = self.task.tip_position()
        p = np.asarray(p_world)
        if self.robot_still():
            return np.linalg.norm(tip - p) < KEEP_OUT_STILL
        return np.linalg.norm(tip[:2] - p[:2]) < ROBOT_KEEP_OUT and -0.10 < p[2] - tip[2] < 0.25

    def path_clear(self, seg) -> bool:
        """The palm path of a segment (and what the hand carries) keeps the keep-out distance from the gripper."""
        tip = self.task.tip_position()
        keep = self._keep_out()
        drop = np.array([0.0, 0.0, 0.5 * self.held.height + HAND_HALF_THICK]) if self.held is not None else None
        for s in np.linspace(0.0, 1.0, 9):
            q = seg.p0 + (seg.p1 - seg.p0) * s
            if np.linalg.norm(q - tip) < keep:
                return False
            if drop is not None and np.linalg.norm(q - drop - tip) < keep + 0.5 * self.held.radius:
                return False
        # the whole arm (and what it carries) along the way must not run into the robot
        keep_palm = self.palm.copy()
        try:
            n = max(5, int(np.ceil(np.linalg.norm(seg.p1 - seg.p0) / 0.04)))      # every 4 cm along the way
            for s in np.linspace(1.0 / n, 1.0, n):
                self._pose_arm(self.to_person(seg.p0 + (seg.p1 - seg.p0) * s))
                parts = list(self.arm_parts) + [self.torso, self.head]        # she leans forward to reach
                if self.held is not None:
                    self._carry(self.to_world(self.palm))
                    parts.append(self.held.shape)
                if any(self.task.robot_overlap(p) is not None for p in parts):
                    return False
            return True
        finally:
            self._pose_arm(keep_palm)
            if self.held is not None:
                self._carry(self.to_world(keep_palm))

    def arm_hits_robot(self, palm_world) -> bool:
        """With the palm there, the arm (or what the hand carries) would touch the robot."""
        keep = self.palm.copy()
        try:
            self._pose_arm(self.to_person(palm_world))
            parts = list(self.arm_parts)
            if self.held is not None:
                self._carry(self.to_world(self.palm))
                parts.append(self.held.shape)
            return any(self.task.robot_overlap(p) is not None for p in parts)
        finally:
            self._pose_arm(keep)
            if self.held is not None:
                self._carry(self.to_world(keep))

    def robot_still(self) -> bool:
        """The gripper moves slower than STILL_SPEED (measured once per step, in update)."""
        return getattr(self, 'tip_speed', 0.0) < STILL_SPEED

    def arm_fits(self, palm_world, ignore=()) -> bool:
        """With the palm there (and what the hand holds with it), the arm and the upper body touch no body of the
        scene but the ignored ones, nor the robot: a careful person does not plough through things to put something
        down."""
        keep = self.palm.copy()
        held = self.held
        ignore = set(ignore)
        if held is not None:
            ignore |= set(held.handles())
        try:
            self._pose_arm(self.to_person(palm_world))
            if held is not None:
                self._carry(self.to_world(self.palm))
                # what the hand carries must not hit anything either
                if self.task.overlaps(held.shape, robot=True, ignore=ignore) is not None:
                    return False
            # the arm and the upper body (leaning forward to reach): the legs are under the table
            for part in self.arm_parts + [self.torso, self.head, self.left_arm]:
                if self.task.overlaps(part, robot=True, ignore=ignore) is not None:
                    return False
            return True
        finally:
            self._pose_arm(keep)
            if held is not None:
                self._carry(self.to_world(keep))

    def gripper_close(self, dist: float) -> bool:
        """The gripper tip is within dist of the hand / forearm surface or of what the hand carries."""
        tip = self.task.tip_position()
        e, h = self.to_world(self.elbow), self.palm_world()
        s = np.clip(np.dot(tip - e, h - e) / max(np.dot(h - e, h - e), 1e-9), 0.0, 1.0)
        if np.linalg.norm(tip - (e + s * (h - e))) < dist + 0.04:
            return True
        if self.held is not None and not self.task.is_grasped(self.held.shape):
            c = self.held.position() + [0.0, 0.0, 0.5 * self.held.height]
            return np.linalg.norm(tip - c) < dist + 0.5 * self.held.radius
        return False

    # --- jobs -------------------------------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.job is not None or bool(self.queue)

    def submit(self, job: Job, priority: bool = False) -> Job:
        """Queues a job; a priority job (the task's) goes first, and interrupts a running preemptible job (a scene
        change) that has not grabbed or brought anything yet."""
        if not priority:
            self.queue.append(job)
            return job
        self.queue.insert(0, job)
        cur = self.job
        if cur is not None and cur.preemptible and not any(s.irreversible for s in cur.steps[:cur.k]):
            cur.status, cur.reason = 'gave_up', 'preempted by the task'
            self.task.log_event('person_interrupts', job=cur.name)
            self.job = None
        return job

    def fail_job(self, reason: str) -> None:
        if self.job is not None:
            self.job.status, self.job.reason = 'gave_up', reason
            self.task.log_event('person_gives_up', job=self.job.name, reason=reason)
            self._after_job()

    def _after_job(self) -> None:
        """Next queued job at once (keeping what is in the hand), else back to the lap; whatever is still in the hand
        is first put back where it was taken from."""
        if self.held is not None and self.job is not None and self.job.name == 'get_ready' and not self.queue:
            self.job = None                     # she holds it, ready: no rest, nothing put back
            return
        if self.held is not None:
            if self.job is not None and self.job.name == 'put_back':
                # putting it back failed (its place stays taken): the person puts it down where the hand is
                self.queue.insert(0, Job('put_down_here', self._put_down_here_steps()))
            elif not (self.queue and self.queue[0].name in ('put_back', 'put_down_here')):
                self.queue.insert(0, self.put_back_job(self.held))
        if self.queue:
            self.job = None
            return
        self.job = Job('rest', [Move(self.lap_world, speed=0.5, wait_for_robot=False)])
        self.job.status = 'running'
        self.job.steps[0].start(self, self.t)

    def put_back_job(self, item) -> Job:
        """Puts the item in the hand back where it was taken from (waiting up to 8 s for the place to be free)."""
        frame, grip = self.taken_from.get(item.name, (item.frame(), 'top'))
        bottom = frame[:3, 3]
        palm = self.palm_pinching(item, bottom) if grip == 'pinch' else self.palm_above(bottom[2] + item.height,
                                                                                        bottom[:2])
        drop = palm + [0.0, 0.0, 0.003]
        return Job('put_back', [Move(drop + [0.0, 0.0, 0.10], 0.5),
                                WaitUntil(lambda: self.task.frame_free(item, frame + _lift(0.003)), 8.0,
                                          'place taken'),
                                Move(drop, 0.25, direct=True), LetGo(),
                                Move(drop + [0.0, 0.0, 0.10], 0.25, direct=True, wait_for_robot=False)])

    def _put_down_here_steps(self) -> List[Step]:
        item = self.held
        p = self.palm_world()
        low = np.array([p[0], p[1], self.task.surface_below(item) + item.height + HAND_HALF_THICK + 0.008])
        return [Move(low, 0.25, direct=True), LetGo(),
                Move(low + [0.0, 0.0, 0.10], 0.25, direct=True, wait_for_robot=False)]

    def flinch(self, t: float) -> None:
        job = self.job
        self.task.log_event('person_flinches', job=job.name if job else None)
        self.contacts += 1
        if job is not None and job.name != 'rest':
            job.attempts += 1
            if job.attempts >= MAX_ATTEMPTS:
                job.status, job.reason = 'gave_up', 'robot contact'
                self.task.log_event('person_gives_up', job=job.name, reason='robot contact')
                job = None
            else:
                # resume after the last irreversible step
                last = max([i for i in range(job.k) if job.steps[i].irreversible], default=-1)
                job.k = last + 1
                self.queue.insert(0, job)
        # back towards the body (what is in the hand stays in the hand: an abandoned job puts it back afterwards)
        steps = [Move(self._retreat_point(), speed=0.9, direct=True, wait_for_robot=False)]
        if (job is None or job.name == 'rest') and self.held is not None:
            self.queue.insert(0, self.put_back_job(self.held))
        self.job = Job('flinch', steps + [Hold(FLINCH_PAUSE)])
        self.job.status = 'running'
        self.job.steps[0].start(self, t)

    def _retreat_point(self) -> np.ndarray:
        """Where the hand withdraws to after a contact: up and back towards the body (over the table edge)."""
        p = self.palm_world()
        q = self.to_person(p)
        q[1] = min(q[1], 0.30)
        w = self.to_world(q)
        w[2] = max(p[2], TABLE_Z + 0.15)
        return w

    def update(self, t: float) -> None:
        self.dt = t - getattr(self, 't', 0.0) if t > getattr(self, 't', 0.0) else 0.05
        self.t = t
        tip = self.task.tip_position()
        prev = getattr(self, '_tip_prev', None)
        self.tip_speed = 0.0 if prev is None else float(np.linalg.norm(tip - prev)) / max(self.dt, 1e-6)
        self._tip_prev = tip
        if self.hidden:
            return
        if self.releasing:
            self._settle_released()
        if self.job is not None and self.job.name not in ('flinch',) and self.touched_by_robot():
            self.flinch(t)
        if self.job is None and self.queue:
            self.job = self.queue.pop(0)
            if self.job.status == 'pending':
                self.task.log_event('person_starts', job=self.job.name)
            self.job.status = 'running'
            self.job.steps[self.job.k].start(self, t)
        job = self.job
        if job is None:
            return
        self._run_job(job, t)
        # the hand may have run into the robot while moving: flinch at once
        if self.job is not None and self.job.name != 'flinch' and self.touched_by_robot():
            self.flinch(t)

    def _run_job(self, job, t: float) -> None:
        while job.k < len(job.steps):
            if not job.steps[job.k].run(self, t):
                break
            if job.status == 'gave_up' or self.job is not job:
                return
            job.k += 1
            if job.k < len(job.steps):
                job.steps[job.k].start(self, t)
        if job.k >= len(job.steps) and self.job is job:
            if job.name not in ('rest', 'flinch'):
                job.status = 'done'
                self.task.log_event('person_done', job=job.name)
                self._after_job()
            elif job.name == 'flinch' and not self.queue:
                self._after_job()          # back to the lap
            else:
                self.job = None

    def plan(self, horizon: float = 3.0, dt: float = 0.1) -> dict:
        """Palm positions the script is about to produce over the horizon, assuming no waiting: the rest of the
        current move, then the following moves of the job (a hold keeps the palm still; a condition ends the plan)."""
        t0 = getattr(self, 't', 0.0)
        times, pts = [], []
        job = self.job
        if job is None or self.hidden:
            return {'t': [round(t0, 3)], 'palm': [[round(float(x), 4) for x in self.palm_target]], 'complete': True}
        segs: List[tuple] = []          # (segment, time already spent in it)
        cur = self.palm_target.copy()
        complete = True
        for k in range(job.k, len(job.steps)):
            step = job.steps[k]
            if isinstance(step, Move):
                if k == job.k and hasattr(step, 'segments'):
                    for i in range(step.k, len(step.segments)):
                        seg = step.segments[i]
                        spent = (t0 - step.t0) if (i == step.k and step.started) else 0.0
                        segs.append((seg, spent))
                    if step.segments:
                        cur = step.segments[-1].p1
                else:
                    try:
                        target = np.asarray(step.target_fn(), dtype=np.float64)
                    except Exception:
                        complete = False
                        break
                    for seg in self.path_segments(cur, target, step.speed, step.cruise, step.direct):
                        segs.append((seg, 0.0))
                    cur = target
            elif isinstance(step, Hold) and np.isfinite(step.duration) and step.until is None:
                spent = (t0 - step.t0) if k == job.k and hasattr(step, 't0') else 0.0
                segs.append((motion.Hold(cur, step.duration), spent))
            elif isinstance(step, (Hold, WaitUntil)):
                complete = False
                break
        tau, t = 0.0, t0
        for seg, spent in segs:
            remaining = seg.duration - spent
            local = spent
            while local < seg.duration and t - t0 <= horizon + 1e-9:
                times.append(round(t, 3))
                pts.append([round(float(x), 4) for x in seg.at(local)])
                local += dt
                t += dt
            if t - t0 > horizon:
                break
        if not times:
            times, pts = [round(t0, 3)], [[round(float(x), 4) for x in cur]]
        return {'t': times, 'palm': pts, 'complete': complete}

    def touched_by_robot(self) -> bool:
        return any(self.task.monitor.robot_touches(p.get_handle()) for p in self.arm_parts + [self.torso, self.head])

    # --- visibility / state -----------------------------------------------------------------------------------------

    def reset(self) -> None:
        if self.held is not None and self.held.state == 'carried':
            self.held.make_passive()
        self.job, self.queue, self.held = None, [], None
        self.releasing = []
        self.taken_from = {}
        self._tip_prev = None
        self.tip_speed = 0.0
        self.contacts = 0
        self.t = 0.0
        self.palm_target = self.lap_world.copy() if hasattr(self, 'lap_world') else self.palm_target
        self.n = self.n_rest.copy()

    def show(self) -> None:
        for p in self.parts:
            p.set_respondable(True)
            p.set_collidable(True)
        for m in self.meshes + [s for s in self.chair.get_objects_in_tree(object_type=ObjectType.SHAPE)]:
            m.set_renderable(True)
        for s in self.chair.get_objects_in_tree(object_type=ObjectType.SHAPE):
            s.set_collidable(True)
        self.hidden = False
        self.reset()
        self.place(-1, 0.3)          # a fixed seat until the task chooses one: nothing depends on the previous episode

    def hide(self) -> None:
        for p in self.parts:
            p.set_respondable(False)
            p.set_collidable(False)
        for m in self.meshes + [s for s in self.chair.get_objects_in_tree(object_type=ObjectType.SHAPE)]:
            m.set_renderable(False)
            m.set_collidable(False)
        far = np.eye(4)
        far[:3, 3] = [-3.0, 3.0, -2.0]
        self.root.set_matrix(far)
        self.hidden = True
        self.job, self.queue, self.held = None, [], None
        self.releasing = []

    def handles(self) -> List[int]:
        """Collision shapes and meshes of the person (not the chair)."""
        return [p.get_handle() for p in self.parts] + [m.get_handle() for m in self.meshes]

    def chair_handles(self) -> List[int]:
        return [s.get_handle() for s in self.chair.get_objects_in_tree(object_type=ObjectType.SHAPE)]

    def part_poses(self) -> Dict[str, np.ndarray]:
        """World pose (position + quaternion) of every collision shape: ground truth of the agent's motion."""
        out = {}
        for p in self.parts:
            m = p.get_matrix()
            out[p.get_name()] = np.r_[m[:3, 3], matrix_to_quat(m)]
        return out

    def remove(self) -> None:
        remove_tree(self.root)
