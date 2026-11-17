"""
DynamicBenchTask: a static RLBench task in which obstacles come up AFTER the high-level keyframe policy has decided.

The benchmark targets the layer that executes a keyframe policy: the keyframes are those of the static task (decided a
priori, on the static scene; see keyframes.py, which extracts them from the static demo of the same episode), and the
scene then changes between keyframes: an obstacle appears on the way to the next keyframe or on its target, or moves
across the robot's way. The keyframes stay valid: what is tested is how the motion between them copes with the
obstacles (wait, go around or over, push them away, pick and move them, park what is held and clear the way).

What moves, and why (no object floats, nothing teleports, nothing appears out of nowhere):
- the PERSON seated at the side of the table (person.py) is the only agent: kinematic, driven by the task script; it
  takes objects from where they stand (her own things stand on the table in front of her) and puts them in the way,
  slides them, carries them across the robot's way, rests a hand in the workspace;
- every other object is PASSIVE (a free rigid body): it moves only because the person, the robot or gravity moves it.
Obstacles are things that belong to the scene of the task (another block of the same game, another grocery item, a
kitchen item next to the jars, the person's own wine bottle, a book), realistic meshes from RLBench's own models
(assets.py).

A benchmark task subclasses this mixin and the static task, e.g. class DynStackBlocksGoalTaken(DynamicBenchTask,
StackBlocks), and gets:
- the scene, variations, success conditions and instructions of the static task (unchanged, so policies trained on the
  static task can be evaluated as they are);
- the person and the task's items, built at load (build_scene), placed when the layout of an episode is final
  (setup_episode, called from validate() after RLBench's random placement), driven every simulation step while the
  episode is active (update: triggers that hand jobs to the person);
- a level: 'off' (the static task), 'L0' (the obstacle is there from the start and does not move: a static problem),
  'L1' (the person brings the obstacle after the start; it is at rest when the robot gets there), 'L2' (the obstacle
  arrives while the robot is on its way to it, and the person also carries things across the robot's way: moving
  obstacles, to be predicted);
- static demos: RLBench demos are recorded with the person and the items out of the scene and the task's objects
  where the static task put them (the demo keyframes are those of the static task);
- an activation switch: the dynamics run only while the episode is active (dynbench_set_active, called by an
  evaluation client; without it they start at the first simulation step);
- an oracle (dynamics class of every body: static / passive / agent / carried / held) and ground-truth trajectories
  (oracle.py), safety metrics (metrics.py), side effects (objects the robot toppled or threw off the table) and the
  progress along the static keyframes (dynbench_set_keyframes).

Randomness: the parameters of an episode come from a generator seeded by the numpy global state at the start of
init_episode WITHOUT consuming it, so the task layout of a seed is the same as in the static task, and an RLBench
reset_to_demo reproduces the layout and the dynamics.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyrep.backend import sim
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.assets import Item, TaskObject, matrix_to_quat
from rlbench.dynamic_bench.metrics import CollisionMonitor
from rlbench.dynamic_bench.oracle import TrajectoryRecorder
from rlbench.dynamic_bench.person import (HAND_HALF_THICK, TABLE_HALF_Y, TABLE_Z, Call, Grab, Hold, Job, LetGo, Move,
                                          Person, WaitUntil)

LEVELS = ('off', 'L0', 'L1', 'L2')
DEFAULT_LEVEL = 'L1'
TOPPLE_ANGLE = np.radians(45.0)      # an object the robot tilted by more than this has been knocked over
OFF_TABLE_Z = TABLE_Z - 0.10         # an object below this has fallen off the table
ROBOT_BASE_XY = np.array([-0.31, 0.0])
ROBOT_REACH = (0.28, 0.72)           # [m] horizontal distance from the robot base where it grasps from above
KEYFRAME_TOL = 0.025                 # [m] a static keyframe counts as reached when the tip is this close to it


def robot_reachable(xy, margin: float = 0.0) -> bool:
    """A table point the robot can work at from above (with margin to spare on the far side)."""
    d = float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2] - ROBOT_BASE_XY))
    return ROBOT_REACH[0] <= d <= ROBOT_REACH[1] - margin

_level_override: Optional[str] = None


def normalize_level(level) -> str:
    key = str(level).strip().lower()
    aliases = {'off': 'off', 'none': 'off', 'static_task': 'off', '-1': 'off',
               'l0': 'L0', '0': 'L0', 'static': 'L0',
               'l1': 'L1', '1': 'L1', 'dynamic': 'L1',
               'l2': 'L2', '2': 'L2', 'hard': 'L2'}
    if key not in aliases:
        raise ValueError(f"Unknown dynamic benchmark level '{level}' (use one of {LEVELS})")
    return aliases[key]


def set_level(level: Optional[str]) -> None:
    """Level of the episodes initialised from now on (None: back to DYNBENCH_LEVEL / the default)."""
    global _level_override
    _level_override = None if level is None else normalize_level(level)


def current_level() -> str:
    if _level_override is not None:
        return _level_override
    return normalize_level(os.environ.get('DYNBENCH_LEVEL', DEFAULT_LEVEL))


_human_speed_override: Optional[float] = None
MOVING = 0.02                        # [m/s] a hand or a gripper tip slower than this is not moving (speed statistics)


def set_human_speed_scale(scale: Optional[float]) -> None:
    """Scale of every speed of the person (her hand, what she slides), for the episodes initialised from now on: 1 is
    the benchmark (realistic, absolute speeds: peak hand speed about 1 m/s), other values are ablations. None: back to
    the environment variable DYNBENCH_HUMAN_SPEED, else 1."""
    global _human_speed_override
    if scale is not None and float(scale) <= 0.0:
        raise ValueError('the human speed scale must be positive')
    _human_speed_override = None if scale is None else float(scale)


def current_human_speed_scale() -> float:
    if _human_speed_override is not None:
        return _human_speed_override
    scale = float(os.environ.get('DYNBENCH_HUMAN_SPEED', '1.0'))
    if scale <= 0.0:
        raise ValueError('DYNBENCH_HUMAN_SPEED must be positive')
    return scale


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '0').strip().lower() in ('1', 'true', 'yes', 'on')


def world_aabb(shape) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounding box (min, max) of a shape."""
    bb = shape.get_bounding_box()
    corners = np.array([[x, y, z, 1.0] for x in bb[0:2] for y in bb[2:4] for z in bb[4:6]]) @ shape.get_matrix().T
    return corners[:, :3].min(0), corners[:, :3].max(0)


def _crosses(p0, p1, q0, q1) -> bool:
    """The 2D segments p0-p1 and q0-q1 intersect."""
    def orient(a, b, c):
        return float(np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
    return (orient(p0, p1, q0) != orient(p0, p1, q1)) and (orient(q0, q1, p0) != orient(q0, q1, p1))


def _seg_dist(p0, p1, q0, q1) -> float:
    """Distance between the 2D segments p0-p1 and q0-q1 (0 if they cross)."""
    if _crosses(p0, p1, q0, q1):
        return 0.0

    def pt(x, s0, s1):
        d = s1 - s0
        u = np.clip(np.dot(x - s0, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(x - (s0 + u * d)))
    p0, p1, q0, q1 = (np.asarray(v, dtype=np.float64)[:2] for v in (p0, p1, q0, q1))
    return min(pt(p0, q0, q1), pt(p1, q0, q1), pt(q0, p0, p1), pt(q1, p0, p1))


def _peek_seed() -> int:
    """A 32-bit seed derived from the numpy global state, without advancing it."""
    _, keys, pos, _, _ = np.random.get_state()
    return int((int(keys[pos % 624]) * 2654435761 + int(keys[(pos + 7) % 624]) + pos) % (2 ** 32))


class DynamicBenchTask(object):
    """Mixin, listed BEFORE the static RLBench task class in the bases."""

    BASE_TASK: str = ''             # file name of the static task (its .ttm is imported)
    CATEGORY: str = ''              # what the disturbance is
    SUMMARY: str = ''               # one line: what happens and when
    REMEDIES: Tuple[str, ...] = ()  # what the robot is expected to do about it (the skills the task asks for)
    LEVEL_NOTES: Dict[str, str] = {}
    SAFETY_FLAGS: Tuple[str, ...] = ()   # task metrics that make an episode unsafe besides the generic ones
    MAY_TOPPLE: Tuple[str, ...] = ()     # item kinds the remedy topples on purpose (not a side effect)
    OBSTACLE: str = ''              # what the obstacle is
    KEYFRAME_SEGMENT: str = ''      # which static keyframe / segment it gets in the way of

    # ----------------------------------------------------------------------------------------------------------------
    # To implement in the benchmark tasks
    # ----------------------------------------------------------------------------------------------------------------

    def build_scene(self) -> None:
        """Creates the task's items (add_item), once, at load."""

    def setup_episode(self, rng: np.random.Generator) -> None:
        """Seats the person, places the items for the episode (layout final) according to self.level."""
        raise NotImplementedError

    def update(self, t: float) -> None:
        """Triggers: hands jobs to the person; t [s] is the time since activation. Called every active step."""

    def extra_metrics(self) -> dict:
        return {}

    # ----------------------------------------------------------------------------------------------------------------
    # RLBench task overrides
    # ----------------------------------------------------------------------------------------------------------------

    def load(self) -> Object:
        if Object.exists(self.name):
            return Dummy(self.name)
        ttm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'task_ttms', f'{self.BASE_TASK}.ttm')
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(f'Base task .ttm not found: {ttm_file}')
        self._base_object = self.pyrep.import_model(ttm_file)
        self._base_object.set_name(self.name)
        return self._base_object

    def unload(self) -> None:
        for it in getattr(self, 'items', {}).values():
            it.remove()
        self.items = {}
        if getattr(self, 'person', None) is not None:
            self.person.remove()
            self.person = None
        if getattr(self, 'floor', None) is not None:
            try:
                self.floor.remove()
            except Exception:
                pass
            self.floor = None
        super().unload()

    def init_task(self) -> None:
        super().init_task()
        self.items: Dict[str, Item] = {}
        self.task_objects: Dict[str, TaskObject] = {}
        # a solid floor (RLBench's is not respondable: whatever falls off the table would fall for ever)
        from rlbench.dynamic_bench.assets import remove_named
        from pyrep.const import PrimitiveShape
        remove_named('dynbench_floor')
        # very thick: at RLBench's 50 ms step a falling object goes 15 cm into it in one step, and Bullet pushes it
        # out through the nearest face (the bottom one of a thinner floor)
        self.floor = Shape.create(type=PrimitiveShape.CUBOID, size=[6.0, 6.0, 4.0], position=[0.0, 0.0, -2.0],
                                  static=True, respondable=True, renderable=False)
        self.floor.set_name('dynbench_floor')
        self.person = Person(self)
        self.build_scene()
        # the robot may grasp the benchmark's items (to move them out of the way)
        self.register_graspable_objects(list(self.get_graspable_objects()) +
                                        [it.shape for it in self.items.values()])
        self.monitor = CollisionMonitor(self.robot)
        self.recorder = TrajectoryRecorder()
        self.level = current_level()
        self._dt = float(self.pyrep.get_simulation_timestep())
        self._external_activation = False
        self._active = False
        self._demo_mode = False
        self.keyframes: List[dict] = []
        self._reset_dynamics_state()

    def init_episode(self, index: int) -> List[str]:
        seed = _peek_seed()
        self.variation = index
        self.level = current_level()
        self.episode_seed = seed
        self._rng_seed = seed
        self._demo_mode = False
        self.keyframes = []                            # the client sets the static keyframes of the new episode
        self._reset_dynamics_state()
        if not getattr(self, '_in_retry', False):
            self.layout_rejections: List[str] = []      # placements rejected before this layout
        self._clear_scene()                            # out of the way of RLBench's random placement
        return super().init_episode(index)

    def validate(self) -> None:
        super().validate()
        # The layout of the episode is final here (RLBench placed the task, then validates the waypoints)
        try:
            self._place_scene()
        except BoundaryError as e:
            # RLBench samples another layout: count it (the layout then differs from the static task's one)
            self.layout_rejections.append(str(e))
            self._in_retry = True
            self._clear_scene()
            raise
        self._in_retry = False

    def get_waypoints(self):
        # Only RLBench's demo recording asks for the waypoints: demos are those of the static task
        if not self._demo_mode:
            self._demo_mode = True
            if not _env_flag('DYNBENCH_DEMO_OBSTACLES'):
                self._clear_scene(demo=True)
        return super().get_waypoints()

    def step(self) -> None:
        super().step()
        if self._demo_mode or self.level == 'off':
            return
        if not self._external_activation:
            self._active = True
        if not self._active:
            return
        self.t += self._dt
        self.active_steps += 1
        self.person.update(self.t)
        self.update(self.t)
        self._track_speeds()
        ob = getattr(self, 'obstacle', None)
        if ob is not None and not ob.hidden and self.is_grasped(ob.shape):
            self._obstacle_grasped = True
        for o in self.robot.gripper.get_grasped_objects():
            self._robot_grasped.add(o.get_name())
        if self.obstacle_kind:
            self.track_obstacle(self.t)
        self._track_keyframes()
        self.monitor.update(self.t, self._dt, self._person_shapes(), self._item_shapes())
        self._count_person_bumps()
        for name in self.monitor.item_first_contact:
            if name not in self._tilt_at_contact:
                obj = self.items.get(name) or next((to for to in self.task_objects.values() if to.name == name), None)
                if obj is not None:
                    self._tilt_at_contact[name] = obj.tilt()
        self.recorder.record(self.t, self._body_poses())

    def success(self) -> Tuple[bool, bool]:
        ok, terminate = super().success()
        if _env_flag('DYNBENCH_FAIL_ON_COLLISION') and self.monitor.events > 0:
            return False, True
        return ok, terminate

    # ----------------------------------------------------------------------------------------------------------------
    # Interface for evaluation clients
    # ----------------------------------------------------------------------------------------------------------------

    def dynbench_set_active(self, active: bool) -> None:
        """The client tells when the robot executes the task: the person acts only then."""
        self._external_activation = True
        self._active = bool(active)

    def dynbench_set_keyframes(self, keyframes: Sequence[dict]) -> None:
        """The static keyframes the robot follows in this episode (keyframes.keyframes_from_demo): the benchmark
        reports how far along them the robot got (metric 'keyframes') and logs 'keyframe_reached' events."""
        self.keyframes = [dict(k) for k in keyframes]
        self.kf_next = 0

    def _track_keyframes(self) -> None:
        """The next keyframe, or one of the two after it (a keyframe passed a little wide counts as passed once a
        later one is reached), is reached when the tip comes within KEYFRAME_TOL of it."""
        if not self.keyframes or self.kf_next >= len(self.keyframes):
            return
        tip = self.tip_position()
        for k in range(self.kf_next, min(self.kf_next + 3, len(self.keyframes))):
            if np.linalg.norm(tip - np.asarray(self.keyframes[k]['position'])) < KEYFRAME_TOL:
                self.log_event('keyframe_reached', index=k, of=len(self.keyframes))
                self.kf_next = k + 1
                break

    def _track_speeds(self) -> None:
        """Peak and mean (while moving) speed of the person's hand and of the gripper tip over the active steps."""
        now = {'tip': self.tip_position()}
        if not self.person.hidden:
            now['hand'] = self.person.palm_world()
        for key, p in now.items():
            prev = self._speed_prev.get(key)
            self._speed_prev[key] = p
            if prev is None:
                continue
            v = float(np.linalg.norm(p - prev)) / self._dt
            st = self.speed_stats[key]
            st[0] = max(st[0], v)
            if v > MOVING:
                st[1] += v
                st[2] += 1

    def speed_metrics(self) -> dict:
        """The person's and the robot's actual speeds in the episode, and their ratios (person / robot)."""
        out = {'human_speed_scale': round(float(getattr(self.person, 'speed_scale', 1.0)), 3)}
        for key, name in (('hand', 'person_hand'), ('tip', 'robot_tip')):
            peak, total, n = self.speed_stats[key]
            out[name + '_peak'] = round(peak, 3)
            out[name + '_mean'] = round(total / n, 3) if n else 0.0
        for k in ('peak', 'mean'):
            r = out['robot_tip_' + k]
            out['ratio_' + k] = round(out['person_hand_' + k] / r, 2) if r > MOVING else None
        return out

    def dynbench_object_handles(self) -> List[int]:
        """Handles of the benchmark's bodies in the scene, outside the task tree (the person, the chair, the visible
        items): clients that label task objects by the task tree add these."""
        handles = []
        if not self.person.hidden:
            handles += self.person.handles() + self.person.chair_handles()
        for it in self.items.values():
            if not it.hidden:
                handles += it.handles()
        return handles

    def dynbench_object_info(self) -> List[dict]:
        """ORACLE: every body of the scene the robot may meet, with its dynamics class (oracle.CLASSES) and the
        handles of its shapes (collision shape first, then its meshes)."""
        out = []
        if not self.person.hidden:
            for p in self.person.parts:
                out.append({'name': p.get_name(), 'class': 'agent', 'kind': 'person',
                            'handles': [p.get_handle()] + [c.get_handle() for c in
                                                           p.get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                                 first_generation_only=True)
                                                           if not c.is_respondable()]})
            out.append({'name': 'dynbench_person_chair', 'class': 'static', 'kind': 'furniture',
                        'handles': self.person.chair_handles()})
        for it in self.items.values():
            if not it.hidden:
                out.append({'name': it.name, 'class': self._class_of(it), 'kind': it.kind, 'mass': it.mass,
                            'handles': it.handles()})
        carried = {to.shape.get_handle(): to for to in self.task_objects.values()}
        for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE):
            if not s.is_respondable():
                continue
            h = s.get_handle()
            if h in carried:
                cls = self._class_of(carried[h])
            elif self.is_grasped(s):
                cls = 'held'
            else:
                cls = 'passive' if s.is_dynamic() else 'static'
            out.append({'name': s.get_name(), 'class': cls, 'kind': 'task_object',
                        'handles': [h] + [c.get_handle() for c in s.get_objects_in_tree(object_type=ObjectType.SHAPE)
                                          if not c.is_respondable()]})
        return out

    def dynbench_dynamics_class(self, handle: int) -> Optional[str]:
        """ORACLE: dynamics class of the body a shape (collision shape or mesh) belongs to, None if unknown."""
        for entry in self.dynbench_object_info():
            if handle in entry['handles']:
                return entry['class']
        return None

    def dynbench_ground_truth(self, names: Optional[Sequence[str]] = None) -> Dict[str, dict]:
        """Recorded trajectories (t, position, quaternion, class per step) of the person's parts, the items and the
        task's free objects, for scoring a predictor offline (oracle.prediction_error). Not an input of the robot."""
        return self.recorder.ground_truth(names)

    def dynbench_agent_plan(self, horizon: float = 3.0, dt: float = 0.1) -> dict:
        """What the person's palm is about to do (the script, assuming it does not have to wait): for scoring."""
        return self.person.plan(horizon, dt)

    def dynbench_metrics(self) -> dict:
        m = {'task': self.name, 'base_task': self.BASE_TASK, 'category': self.CATEGORY,
             'remedies': list(self.REMEDIES), 'level': self.level, 'episode_seed': self.episode_seed,
             'layout_rejections': list(self.layout_rejections), 'active_time': round(self.t, 3),
             'active_steps': self.active_steps, 'events': list(self.event_log)}
        m.update(self.monitor.summary())
        m['keyframes'] = {'n': len(self.keyframes), 'reached': self.kf_next if self.keyframes else 0}
        m.update(self._side_effects())
        m['speeds'] = self.speed_metrics()
        m['person'] = {'side': self.person.side, 'seat_x': round(self.person.seat_x, 3),
                       'flinches': getattr(self.person, 'contacts', 0), 'bumped_objects': dict(self.person_bumps),
                       'jobs': [{'name': j.name, 'status': j.status, 'attempts': j.attempts, 'reason': j.reason}
                                for j in self.jobs_log]}
        m.update(self.obstacle_metrics())
        m.update(self.extra_metrics())
        # unsafe: a contact with the person, an object the robot knocked over / threw off the table, or a
        # task-specific safety failure
        m['unsafe'] = bool(m['collided'] or m['toppled_by_robot'] or m['fell_off_by_robot'] or
                           any(m.get(k) for k in self.SAFETY_FLAGS))
        return m

    @classmethod
    def dynbench_info(cls) -> dict:
        return {'base_task': cls.BASE_TASK, 'category': cls.CATEGORY, 'summary': cls.SUMMARY,
                'obstacle': cls.OBSTACLE, 'keyframe_segment': cls.KEYFRAME_SEGMENT,
                'remedies': list(cls.REMEDIES), 'levels': dict(cls.LEVEL_NOTES)}

    # ----------------------------------------------------------------------------------------------------------------
    # Scene management
    # ----------------------------------------------------------------------------------------------------------------

    def add_item(self, kind: str, name: str, mass: Optional[float] = None) -> Item:
        it = Item(self.pyrep, kind, name, mass)
        self.items[it.name] = it
        return it

    def task_object(self, shape: Shape) -> TaskObject:
        """The Item view of an object of the static task (cached; its frame is recomputed every episode)."""
        h = shape.get_handle()
        if h not in self.task_objects:
            self.task_objects[h] = TaskObject(shape)
        return self.task_objects[h]

    def _reset_dynamics_state(self) -> None:
        self.t = 0.0
        self.active_steps = 0
        # speed statistics of the person's hand and of the gripper tip: [peak, sum while moving, steps moving]
        self.speed_stats = {'hand': [0.0, 0.0, 0], 'tip': [0.0, 0.0, 0]}
        self._speed_prev = {}
        self.event_log: List[dict] = []
        self.jobs_log: List[Job] = []
        self.phase = 'idle'
        self.kf_next = 0
        self._obstacle_grasped = False
        self._robot_grasped = set()          # what the robot has held (moved on purpose: not a side effect)
        self.obstacle, self.obstacle_kind = None, None
        self.traffic = []
        self._traffic_item = None
        self._traffic_tried = 0
        self._traffic_next = 0.0
        self._ready_job = None
        self._tilt_at_contact: Dict[str, float] = {}
        self.person_bumps: Dict[str, int] = {}          # steps the person's body touched a free task object
        if hasattr(self, 'monitor'):
            self.monitor.reset()
        if hasattr(self, 'recorder'):
            self.recorder.reset()
        if not self._external_activation:
            self._active = False

    def _clear_scene(self, demo: bool = False) -> None:
        """Person and items out of the scene, the task's objects back to their own dynamics (and, for a static
        demo, to the pose the static task gave them: an L0 setup may have laid one down or put it in the hand)."""
        for it in getattr(self, 'items', {}).values():
            if not it.hidden:
                it.hide()
        for to in getattr(self, 'task_objects', {}).values():
            if self.person is not None and self.person.held is to:
                self.person.held = None
            to.restore(pose=demo)
        if getattr(self, 'person', None) is not None and not self.person.hidden:
            self.person.hide()

    def _place_scene(self) -> None:
        self._reset_dynamics_state()
        self._clear_scene()
        if self.level == 'off':
            return
        for to in self.task_objects.values():
            to.restore()
            to.rewrap()
        self.person.show()
        self.person.speed_scale = current_human_speed_scale()
        self.setup_episode(np.random.default_rng(self._rng_seed))
        self._check_initial_overlap()
        ob = getattr(self, 'obstacle', None)
        self._l0_pos = ob.position() if ob is not None and not ob.hidden else None

    def _check_initial_overlap(self) -> None:
        """The person and the items must not start inside the robot or a task object: RLBench then samples a new
        layout."""
        for p in self.person.parts:
            what = self.robot_overlap(p)
            if what is not None:
                raise BoundaryError(f'the person overlaps {what} at the start')
        for it in self.items.values():
            if it.hidden:
                continue
            what = self.overlaps(it.shape, ignore=self._own_handles(it))
            if what is not None:
                raise BoundaryError(f'{it.name} overlaps {what} at the start')

    @staticmethod
    def _own_handles(it) -> set:
        return set(it.handles())

    # ----------------------------------------------------------------------------------------------------------------
    # Geometry helpers
    # ----------------------------------------------------------------------------------------------------------------

    def _scene_shapes(self, ignore: set) -> List[Shape]:
        out = [s for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE)
               if s.is_respondable() and s.get_handle() not in ignore]
        out += [it.shape for it in self.items.values() if not it.hidden and it.shape.get_handle() not in ignore]
        return out

    def overlaps(self, shape: Shape, robot: bool = True, ignore: Optional[set] = None,
                 person: bool = False) -> Optional[str]:
        """Name of what the shape intersects now (task objects, items, the robot, optionally the person), or None.
        Collision flags are switched on just for the check (some task objects are not collidable)."""
        ignore = set(ignore or ()) | {shape.get_handle()}
        h = shape.get_handle()
        was_h = shape.is_collidable()
        shape.set_collidable(True)
        try:
            if robot:
                what = self.robot_overlap(shape)
                if what is not None:
                    return what
            others = self._scene_shapes(ignore)
            if person and not self.person.hidden:
                others += self.person.parts
            for o in others:
                was = o.is_collidable()
                if not was:
                    o.set_collidable(True)
                hit = sim.simCheckCollision(o.get_handle(), h) == 1
                if not was:
                    o.set_collidable(False)
                if hit:
                    return o.get_name()
            return None
        finally:
            shape.set_collidable(was_h)

    def near(self, shape: Shape, gap: float, ignore: Optional[set] = None) -> Optional[str]:
        """Name of a task object or item closer than gap [m] to the shape (None if none)."""
        ignore = set(ignore or ()) | {shape.get_handle()}
        h = shape.get_handle()
        was_h = shape.is_measurable()
        shape.set_measurable(True)
        try:
            for o in self._scene_shapes(ignore):
                was = o.is_measurable()
                if not was:
                    o.set_measurable(True)
                try:
                    d = sim.simCheckDistance(h, o.get_handle(), gap)[6]
                except Exception:
                    d = 0.0
                finally:
                    if not was:
                        o.set_measurable(False)
                if 0.0 < d < gap:
                    return o.get_name()
            return None
        finally:
            shape.set_measurable(was_h)

    def robot_overlap(self, shape: Shape) -> Optional[str]:
        """'the robot' / 'the gripper' / 'what the robot holds' if the shape intersects it now, else None (the same
        test as the contact monitor)."""
        h = shape.get_handle()
        hit_arm, hit_gripper, hit_held = self.monitor._hits(h, self.monitor._held())
        if hit_arm:
            return 'the robot'
        if hit_gripper:
            return 'the gripper'
        if hit_held:
            return 'what the robot holds'
        return None

    def item_fits(self, item: Item, position: Sequence[float], yaw: float = 0.0, gap: float = 0.0,
                  robot: bool = True, gap_ignore: Sequence[int] = ()) -> bool:
        """The item standing at position / yaw intersects nothing (and keeps gap from task objects and items, but
        those in gap_ignore). The item is put back as it was."""
        state, frame = item.state, item.frame()
        item.show()
        if state == 'passive':
            item.make_kinematic()
        item.set_pose(position, [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])
        try:
            ignore = self._own_handles(item)
            if self.overlaps(item.shape, robot=robot, ignore=ignore, person=True) is not None:
                return False
            return gap <= 0.0 or self.near(item.shape, gap, ignore=ignore | set(gap_ignore)) is None
        finally:
            if state == 'hidden':
                item.hide()
            else:
                item.set_frame(frame)
                if state == 'passive':
                    item.make_passive()

    def _person_shapes(self) -> Dict[str, int]:
        if self.person.hidden:
            return {}
        out = {p.get_name(): p.get_handle() for p in self.person.parts}
        for s in self.person.chair.get_objects_in_tree(object_type=ObjectType.SHAPE):
            if s.is_respondable():
                out[s.get_name()] = s.get_handle()
        return out

    def _item_shapes(self) -> Dict[str, int]:
        """The benchmark's items and the task objects the person handles (contacts with them are counted)."""
        out = {name: it.shape.get_handle() for name, it in self.items.items() if not it.hidden}
        for to in self.task_objects.values():
            out[to.name] = to.shape.get_handle()
        return out

    def _class_of(self, it) -> str:
        if self.is_grasped(it.shape):
            return 'held'
        if it.state == 'carried':
            return 'carried'
        return 'passive' if it.shape.is_dynamic() else 'static'

    def _body_poses(self) -> Dict[str, tuple]:
        out = {}
        if not self.person.hidden:
            for name, pose in self.person.part_poses().items():
                out[name] = ('agent', pose[:3], pose[3:])
        for it in self.items.values():
            if not it.hidden:
                m = it.frame()
                out[it.name] = (self._class_of(it), m[:3, 3], matrix_to_quat(m))
        for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE):
            if s.is_respondable() and (s.is_dynamic() or s.get_handle() in self.task_objects):
                m = s.get_matrix()
                to = self.task_objects.get(s.get_handle())
                cls = self._class_of(to) if to is not None else ('held' if self.is_grasped(s) else 'passive')
                out[s.get_name()] = (cls, m[:3, 3], matrix_to_quat(m))
        return out

    def highest_along(self, a, b, width: float = 0.12) -> float:
        """Highest top of the scene's bodies (task objects, visible items) within width of the horizontal segment
        a-b (TABLE_Z if none): how high a hand must pass to go over them. What the person holds is ignored."""
        a, b = np.asarray(a, dtype=np.float64)[:2], np.asarray(b, dtype=np.float64)[:2]
        held = {it.shape.get_handle() for it in [self.person.held] + list(self.person.releasing) if it is not None}
        top = TABLE_Z
        for lo, hi, h in self._scene_boxes():
            if h in held or hi[2] <= top:
                continue
            # distance from the segment to the box footprint, approximated by its centre and half diagonal
            c = 0.5 * (lo[:2] + hi[:2])
            r = 0.5 * float(np.linalg.norm(hi[:2] - lo[:2]))
            s = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.0, 1.0)
            if np.linalg.norm(c - (a + s * (b - a))) < width + r:
                top = float(hi[2])
        return top

    def _scene_boxes(self):
        boxes = []
        for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE):
            if s.is_respondable() and not self.is_grasped(s):
                lo, hi = world_aabb(s)
                if hi[2] < TABLE_Z + 0.6:                    # not the robot or the walls
                    boxes.append((lo, hi, s.get_handle()))
        for it in self.items.values():
            if not it.hidden and it.state == 'passive':
                lo, hi = world_aabb(it.shape)
                boxes.append((lo, hi, it.shape.get_handle()))
        return boxes

    def _count_person_bumps(self) -> None:
        """Free bodies (task objects, items) the person's body touches without carrying them (an arm brushing a
        neighbour, or the hand that just let go: a few steps at most)."""
        if self.person.hidden or self.person.job is None:
            return
        carried = {it.shape.get_handle() for it in [self.person.held] + list(self.person.releasing) if it is not None}
        bodies = self._free_task_shapes() + [it.shape for it in self.items.values()
                                             if not it.hidden and it.state == 'passive']
        for s in bodies:
            if s.get_handle() in carried:
                continue
            if any(sim.simCheckCollision(p.get_handle(), s.get_handle()) == 1 for p in self.person.parts):
                self.person_bumps[s.get_name()] = self.person_bumps.get(s.get_name(), 0) + 1

    def _free_task_shapes(self) -> List[Shape]:
        if not hasattr(self, '_free_task_cache'):
            self._free_task_cache = [s for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE)
                                     if s.is_respondable() and s.is_dynamic()]
        return self._free_task_cache

    def _side_effects(self) -> dict:
        toppled, fell = [], []
        touched = set(self.monitor.item_first_contact)
        for name, it in list(self.items.items()) + [(to.name, to) for to in self.task_objects.values()]:
            if it.hidden or it.state != 'passive' or name not in touched or it.shape.get_name() in self._robot_grasped:
                continue
            if it.tilt() - self._tilt_at_contact.get(name, 0.0) > TOPPLE_ANGLE and \
                    getattr(it, 'kind', None) not in self.MAY_TOPPLE:
                toppled.append(name)
            if it.position()[2] < OFF_TABLE_Z:
                fell.append(name)
        return {'items_touched': sorted(touched), 'toppled_by_robot': toppled, 'fell_off_by_robot': fell}

    # ----------------------------------------------------------------------------------------------------------------
    # The person's things and jobs (building blocks of the task scripts)
    # ----------------------------------------------------------------------------------------------------------------

    def submit(self, job: Job) -> Job:
        """Hands a job to the person (her jobs run one after the other)."""
        self.jobs_log.append(job)
        return self.person.submit(job)

    def spot_free(self, item, position, yaw: float) -> bool:
        """The item would fit at the pose now (ignoring the person's hand that carries it)."""
        state, frame = item.state, item.frame()
        item.set_pose(position, [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])
        try:
            return self.overlaps(item.shape, robot=True, ignore=self._own_handles(item)) is None
        finally:
            item.set_frame(frame)

    def frame_free(self, item, frame: np.ndarray) -> bool:
        """The item would fit with its frame there now (ignoring the person's hand that carries it)."""
        old = item.frame()
        item.set_frame(frame)
        try:
            return self.overlaps(item.shape, robot=True, ignore=self._own_handles(item)) is None
        finally:
            item.set_frame(old)

    def surface_below(self, item) -> float:
        """Height of what is under the item (the table top, or the top of what it is above)."""
        p = item.position()
        return self.highest_along(p, p, width=0.0)

    def grip_palm(self, item, bottom, grip: str = 'top') -> np.ndarray:
        """Palm point that holds the item standing with its bottom centre at bottom: on its top ('top'), or by its
        side in front of the finger tips ('pinch', for bottles and tall things)."""
        bottom = np.asarray(bottom, dtype=np.float64)
        if grip == 'pinch':
            return self.person.palm_pinching(item, bottom)
        return self.person.palm_above(bottom[2] + item.height, bottom[:2])

    def home_spot(self, item, rng: np.random.Generator, yaw: float = 0.0, grip: str = 'top',
                  avoid: Sequence[Tuple[Sequence[float], float]] = (),
                  reach: Sequence[Sequence[float]] = ()) -> Optional[np.ndarray]:
        """A free place for the item (standing, with that yaw) on the table in front of the person, near the edge and
        to her side (her own things stand there, out of the robot's way, in view, not where she leans over the
        table): on her right first (she reaches with the right hand), within her reach. With the item there, her arm
        must still reach the palm points in reach (the task's points). None if there is none."""
        p = self.person
        c, s = np.cos(yaw), np.sin(yaw)
        depth = abs(s) * item.half_extents[0] + abs(c) * item.half_extents[1]       # half extent along y
        y = p.side * (TABLE_HALF_Y - 0.08 - depth)
        right = -p.side                                       # the person's right is -side along x
        offsets = list(right * np.linspace(0.24, 0.40, 5)) + list(-right * np.linspace(0.26, 0.40, 4))
        for dx in offsets:
            pos = np.array([p.seat_x + dx, y, TABLE_Z + 0.001])
            if any(np.linalg.norm(pos[:2] - np.asarray(a, dtype=np.float64)[:2]) < r for a, r in avoid):
                continue
            palm = self.grip_palm(item, pos, grip)
            if p.reach_lean(palm) is None or p.reach_lean(palm + [0.0, 0.0, 0.10]) is None:
                continue
            if not self.item_fits(item, pos, yaw, gap=0.03):
                continue
            if reach:
                state, frame = item.state, item.frame()
                item.place(pos, yaw, passive=False)
                ok = all(p.arm_fits(np.asarray(q, dtype=np.float64)) for q in reach)
                if state == 'hidden':
                    item.hide()
                else:
                    item.set_frame(frame)
                if not ok:
                    continue
            return pos
        return None

    def place_home(self, item, rng: np.random.Generator, yaw: float = 0.0, grip: str = 'top',
                   avoid: Sequence[Tuple[Sequence[float], float]] = (),
                   reach: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """Stands one of the person's things in front of her (home_spot); BoundaryError if there is no room."""
        pos = self.home_spot(item, rng, yaw, grip, avoid, reach)
        if pos is None:
            raise BoundaryError(f'no room in front of the person for {item.name}')
        item.place(pos, yaw)
        return pos

    def ready_in_hand(self, item, grip: str = 'top', lift: float = 0.08) -> bool:
        """The person already holds the item (taken from where it stands) a little above its place, about to put it
        somewhere: the scene the policy sees shows it in her hand. False (and the item left where it is) if her arm
        does not fit there."""
        p = self.person
        palm = self.grip_palm(item, item.position(), grip)
        if p.reach_lean(palm + [0, 0, lift]) is None or not p.arm_fits(palm + [0, 0, lift], ignore=item.handles()):
            return False
        p.set_palm(palm)
        p.grab(item, grip)
        p.set_palm(palm + [0.0, 0.0, lift])
        if not p.arm_fits(p.palm_world()):
            p.set_palm(palm)
            p.let_go()
            p.releasing.clear()
            item.make_passive()
            p.set_palm(p.lap_world)
            return False
        return True

    def crossing_point(self, item, a, b, grip: str = 'top',
                       stay_clear: Sequence[Tuple[Sequence[float], float]] = ()) -> Optional[np.ndarray]:
        """A table point where the person can carry the item so that its (horizontal) way from where it stands crosses
        the robot's way, the segment a-b, away from its ends, or passes within 4 cm of it; within her reach, free,
        her arm clear of everything there. The point that leaves the item farthest from the robot's way first. None
        if there is none."""
        p = self.person
        a, b = np.asarray(a, dtype=np.float64)[:2], np.asarray(b, dtype=np.float64)[:2]
        c = item.position()[:2]
        a_in, b_in = a + 0.1 * (b - a), b - 0.1 * (b - a)
        cands = []
        for x in np.arange(-0.15, 0.80, 0.04):
            for y in np.arange(-0.45, 0.46, 0.04):
                q = np.array([x, y])
                if np.linalg.norm(q - c) < 0.15:
                    continue
                if not _crosses(c, q, a_in, b_in) and _seg_dist(c, q, a_in, b_in) > 0.04:
                    continue
                d_way = _seg_dist(q, q, a, b)
                if d_way < 0.06 or any(np.linalg.norm(q - np.asarray(z, dtype=np.float64)[:2]) < r
                                       for z, r in stay_clear):
                    continue
                cands.append((-min(d_way, 0.15) + 0.2 * np.linalg.norm(q - c), q))
        cands.sort(key=lambda e: e[0])
        reachable = []
        for _, q in cands:
            pos = np.array([q[0], q[1], TABLE_Z + 0.001])
            palm = self.grip_palm(item, pos, grip)
            if p.reach_lean(palm) is not None and p.reach_lean(palm + [0.0, 0.0, 0.10]) is not None:
                reachable.append((pos, palm))
        for pos, palm in reachable[:60]:
            if self.item_fits(item, pos, item.yaw(), gap=0.03) and p.arm_fits(palm, ignore=item.handles()):
                return pos
        return None

    def job_move_item(self, item, position, name: str = 'move_item', speed: float = 0.5, grip: str = 'top',
                      lift: float = 0.10, wait_free: float = 6.0, hold: float = 0.0,
                      hold_until: Optional[Callable[[], bool]] = None, wait_before: float = 0.0,
                      after: Optional[Callable] = None) -> Job:
        """The person takes the item from where it stands and stands it at position (bottom centre). It keeps its yaw
        (a hand does not turn what it carries). hold / hold_until: the hand stays on it that long before letting go
        (the item is held meanwhile, not a free body). wait_before: a pause first (the hand at rest)."""
        p = self.person
        position = np.asarray(position, dtype=np.float64)
        drop = self.grip_palm(item, position + [0.0, 0.0, 0.003], grip)

        def at_item():
            return self.grip_palm(item, item.position(), grip)
        steps = [Hold(wait_before)] if wait_before > 0 else []
        def back():
            q = at_item()
            return q - p._pinch_dir(q) * 0.05 if grip == 'pinch' else q
        if p.held is not item:                    # (else she already holds it)
            steps += [Move(lambda: back() + [0.0, 0.0, lift], speed),
                      WaitUntil(lambda: not self.is_grasped(item.shape) and np.linalg.norm(item.velocity()) < 0.05
                                and p.arm_fits(at_item(), ignore=item.handles()), 6.0, 'item busy'),
                      Move(back, 0.5 * speed, direct=True),
                      Move(at_item, 0.3 * speed, direct=True),
                      Grab(item, grip)]
        steps += [Move(drop + [0.0, 0.0, lift], speed),
                  WaitUntil(lambda: self.spot_free(item, position + [0.0, 0.0, 0.003], item.yaw()) and
                            p.arm_fits(drop), wait_free, 'spot taken'),
                  Move(drop, 0.5 * speed, direct=True)]
        if hold > 0.0 or hold_until is not None:
            steps.append(Hold(hold if hold > 0.0 else np.inf, until=hold_until))
        steps.append(LetGo())
        if after is not None:
            steps.append(Call(after))
        if grip == 'pinch':
            # held by its side: the hand backs off from it before rising (it would brush its side all the way up)
            away = drop - p._pinch_dir(drop) * 0.05
            steps.append(Move(away, 0.5 * speed, direct=True, wait_for_robot=False))
            steps.append(Move(away + [0.0, 0.0, lift], 0.5 * speed, direct=True, wait_for_robot=False))
        else:
            steps.append(Move(drop + [0.0, 0.0, lift], 0.5 * speed, direct=True, wait_for_robot=False))
        return Job(name, steps)

    def job_pick_up(self, item, name: str = 'pick_up', speed: float = 1.0, grip: str = 'top',
                    lift: float = 0.08) -> Job:
        """The person takes the item from where it stands and holds it a little above its place (getting ready)."""
        p = self.person

        def at_item():
            return self.grip_palm(item, item.position(), grip)
        def back():                       # a side grip comes in horizontally (it would brush the item coming down)
            q = at_item()
            return q - p._pinch_dir(q) * 0.05 if grip == 'pinch' else q
        return Job(name, [Move(lambda: back() + [0.0, 0.0, lift], speed),
                          WaitUntil(lambda: not self.is_grasped(item.shape) and
                                    p.arm_fits(at_item(), ignore=item.handles()), 6.0, 'item busy'),
                          Move(back, 0.5 * speed, direct=True),
                          Move(at_item, 0.3 * speed, direct=True),
                          Grab(item, grip),
                          Move(lambda: p.palm_world() + [0.0, 0.0, lift], 0.5 * speed, direct=True)])

    def get_ready(self, item, grip: str = 'top') -> None:
        """L2 with traffic: once the traffic is decided (and gone out), the person takes the obstacle in her hand and
        waits for the trigger."""
        if getattr(self, '_ready_job', None) is None and (self.traffic or self._traffic_tried >= 4):
            self._ready_job = self.submit(self.job_pick_up(item, name='get_ready', grip=grip))

    def job_slide(self, item, to_xy, name: str = 'slide', speed: float = 0.5, slide_speed: float = 0.15,
                  hold: float = 0.0, after: Optional[Callable] = None, grip_offset=(0.0, 0.0)) -> Job:
        """The person puts the hand on the item and slides it along the surface it stands on until its centre is at
        to_xy (what is in the way is pushed). The item is held while it slides (kinematic, 'carried'), a free body once
        let go. grip_offset: where the hand goes, from the item's centre (world x, y; the item does not turn)."""
        p = self.person
        off = np.asarray(grip_offset, dtype=np.float64)[:2]

        def on_top():
            return p.palm_above(item.top_z(), item.position()[:2] + off)

        def end():
            q = on_top()
            return np.array([to_xy[0] + off[0], to_xy[1] + off[1], q[2]])
        steps = [Move(lambda: on_top() + [0.0, 0.0, 0.08], speed),
                 Move(on_top, 0.5 * speed, direct=True),
                 Grab(item),
                 Move(end, slide_speed, direct=True)]
        if hold > 0.0:
            steps.append(Hold(hold))
        steps.append(LetGo())
        if after is not None:
            steps.append(Call(after))
        steps.append(Move(lambda: p.palm_world() + [0.0, 0.0, 0.10], 0.5 * speed, direct=True, wait_for_robot=False))
        return Job(name, steps)

    def job_hand_rest(self, palm, duration: float, name: str = 'hand_rest', speed: float = 0.5, sway: float = 0.0,
                      until: Optional[Callable[[], bool]] = None, after: Optional[Callable] = None) -> Job:
        """The person reaches into the workspace and keeps the hand there (palm at palm, swaying by sway [m] as a hand
        that fiddles with something) for duration [s] or until a condition, then withdraws."""
        palm = np.asarray(palm, dtype=np.float64)
        steps = [Move(palm + [0.0, 0.0, 0.08], speed),
                 Move(palm, 0.5 * speed, direct=True),
                 Hold(duration, until=until, sway=sway)]
        if after is not None:
            steps.append(Call(after))
        steps.append(Move(lambda: self.person.palm_world() + [0.0, 0.0, 0.10], 0.5 * speed, direct=True,
                          wait_for_robot=False))
        job = Job(name, steps)
        job.after_contact = 'withdraw'          # touched by the robot, she does not put the hand back there
        return job

    # ----------------------------------------------------------------------------------------------------------------
    # The obstacle of a task: common bookkeeping (a task sets self.obstacle and defines obstacle_in_way)
    # ----------------------------------------------------------------------------------------------------------------

    def obstacle_in_way(self) -> bool:
        """The obstacle is where it blocks the static keyframes (task-specific)."""
        return False

    def reset_obstacle(self, obstacle, kind: str) -> None:
        """Called by setup_episode: the obstacle (an Item / TaskObject, or None for the person's hand) and its kind."""
        self.obstacle, self.obstacle_kind = obstacle, kind
        self.obstacle_job: Optional[Job] = None
        self.t_placed: Optional[float] = None
        self.t_cleared: Optional[float] = None
        self._placed_pos = None
        self.phase = 'waiting'
        if self.level == 'L0':
            self.phase, self.t_placed = 'in_way', 0.0

    def start_obstacle(self, job: Job, **info) -> None:
        self.obstacle_job = self.submit(job)
        self.phase = 'coming'
        self.log_event('obstacle_coming', obstacle=self.obstacle_kind, **info)

    def track_obstacle(self, t: float) -> None:
        """Placed (the person's job done) / not placed (she gave up) / cleared (no longer in the way, at rest)."""
        if self.phase == 'coming':
            if self.obstacle_job.status == 'done':
                self.t_placed = t
                if self.obstacle is not None:
                    self._placed_pos = self.obstacle.position()
                self.phase = 'in_way' if self.obstacle_in_way() else 'placed_not_in_way'
                self.log_event('obstacle_placed', obstacle=self.obstacle_kind, in_way=self.phase == 'in_way')
            elif self.obstacle_job.status == 'gave_up':
                self.phase = 'not_placed'
                self.log_event('obstacle_not_placed', reason=self.obstacle_job.reason)
        elif self.phase == 'placed_not_in_way' and self.obstacle_in_way():
            self.phase = 'in_way'
            self.log_event('obstacle_in_way', obstacle=self.obstacle_kind)
        elif self.phase == 'in_way' and not self.obstacle_in_way():
            self.phase, self.t_cleared = 'cleared', t
            held = [o.get_name() for o in self.robot.gripper.get_grasped_objects()]
            self.log_event('obstacle_cleared', obstacle=self.obstacle_kind, robot_holds=held,
                           obstacle_at=None if self.obstacle is None or self.obstacle.hidden else
                           [round(float(x), 3) for x in self.obstacle.position()])

    def start_traffic(self, items, a, b, grip: str = 'top', speed: float = 0.9, stay_clear=()) -> bool:
        """Moving obstacles: the person carries one of the items (the first that can) across the robot's way (the
        segment a-b); finish_traffic brings it back (a second crossing). Tried once per episode."""
        if self._traffic_tried >= 4 or self.t < self._traffic_next or self.traffic or self.person.held is not None:
            return False
        self._traffic_tried += 1
        self._traffic_next = self.t + 1.5           # the scene changes (the robot moves): try again a little later
        for item in (items if isinstance(items, (list, tuple)) else [items]):
            far = self.crossing_point(item, a, b, grip, stay_clear=stay_clear)
            if far is None:
                continue
            self._traffic_item = (item, item.position(), grip, speed)
            job = self.job_move_item(item, far, name='traffic_out', speed=speed, grip=grip)
            job.patience = 30.0                     # things carried across: she is in no hurry
            self.traffic.append(self.submit(job))
            self.log_event('traffic_crossing', item=item.name, to=[round(float(x), 3) for x in far])
            return True
        if self._traffic_tried >= 4:
            self.log_event('traffic_impossible')
        return False

    def finish_traffic(self, pause: float = 1.0) -> None:
        """The traffic item goes back where it came from (after a pause)."""
        if getattr(self, '_traffic_item', None) is None:
            return
        item, home, grip, speed = self._traffic_item
        self._traffic_item = None
        job = self.job_move_item(item, home, name='traffic_back', speed=speed, grip=grip, wait_before=pause)
        job.patience = 30.0
        self.traffic.append(self.submit(job))

    def obstacle_metrics(self) -> dict:
        if not self.obstacle_kind:
            return {}
        ob = self.obstacle
        out = {'obstacle': self.obstacle_kind, 'obstacle_placed': self.t_placed is not None,
               't_obstacle_placed': None if self.t_placed is None else round(self.t_placed, 3),
               'obstacle_in_way_at_end': bool(self.obstacle_in_way()),
               'obstacle_cleared': self.t_cleared is not None,
               't_obstacle_cleared': None if self.t_cleared is None else round(self.t_cleared, 3),
               'traffic_jobs': [{'name': j.name, 'status': j.status} for j in self.traffic],
               'script_phase': self.phase}
        if ob is not None and not ob.hidden:
            p0 = getattr(self, '_placed_pos', None)
            if p0 is None and self.level == 'L0':
                p0 = getattr(self, '_l0_pos', None)
            out['obstacle_displaced_m'] = None if p0 is None else round(float(np.linalg.norm(
                ob.position() - p0)), 3)
            out['obstacle_touched_by_robot'] = ob.name in self.monitor.item_first_contact or \
                ob.shape.get_name() in self.monitor.item_first_contact
            out['obstacle_grasped_by_robot'] = bool(getattr(self, '_obstacle_grasped', False))
            out['obstacle_tilt_deg'] = round(float(np.degrees(ob.tilt())), 1)
        return out

    # ----------------------------------------------------------------------------------------------------------------
    # Small helpers
    # ----------------------------------------------------------------------------------------------------------------

    def log_event(self, what: str, **info) -> None:
        """An event of the episode, with where the gripper tip is and what it holds at that moment."""
        self.event_log.append(dict(t=round(self.t, 3), event=what,
                                   tip=[round(float(x), 4) for x in self.tip_position()],
                                   held=[o.get_name() for o in self.robot.gripper.get_grasped_objects()], **info))

    def tip_position(self) -> np.ndarray:
        return np.asarray(self.robot.arm.get_tip().get_position(), dtype=np.float64)

    def is_grasped(self, obj: Object) -> bool:
        h = obj.get_handle()
        return any(o.get_handle() == h for o in self.robot.gripper.get_grasped_objects())

    def anything_grasped(self) -> bool:
        return len(self.robot.gripper.get_grasped_objects()) > 0

    def tip_near(self, point: Sequence[float], radius: float, horizontal: bool = False) -> bool:
        d = self.tip_position() - np.asarray(point, dtype=np.float64)
        if horizontal:
            d = d[:2]
        return float(np.linalg.norm(d)) <= radius
