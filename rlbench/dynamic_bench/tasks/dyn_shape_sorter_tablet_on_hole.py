import numpy as np
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.assets import set_friction
from rlbench.dynamic_bench.core import DynamicBenchTask, world_aabb
from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES, PlaceShapeInShapeSorter

SORTER_FRICTION = 0.3        # the sorter's hard plastic (RLBench leaves it at a rubber-like 0.71 / 0.5)


class DynShapeSorterTabletOnHole(DynamicBenchTask, PlaceShapeInShapeSorter):
    """
    place_shape_in_shape_sorter: after the keyframes have been decided, the person lays her tablet flat on the sorter,
    over the target hole: the static release keyframe is now above the tablet. The tablet is too wide for the gripper
    and too thin (7 mm) to be pushed by its edge: it has to be DRAGGED off the hole, the closed gripper pressing on it
    from above (the fingers grip it better than the sorter's plastic holds it).
    """

    BASE_TASK = 'place_shape_in_shape_sorter'
    CATEGORY = 'keyframe target covered (drag)'
    OBSTACLE = 'the person\'s tablet, lying flat on the sorter over the target hole'
    KEYFRAME_SEGMENT = 'above the hole -> release'
    SUMMARY = 'the person lays her tablet on the sorter, over the target hole'
    REMEDIES = ('drag the tablet off the hole: press on it from above with the closed gripper and move (it is too '
                'wide to grasp and too thin to be pushed by its edge); park the shape first',
                'L2: wait until the person takes her hand off the tablet',
                'then insert the shape')
    LEVEL_NOTES = {
        'L0': 'the tablet lies on the sorter over the hole from the start',
        'L1': 'the person holds her tablet (just picked up from her side of the table) and, right after the start, '
              'lays it over the hole (peak speed 1 m/s)',
        'L2': 'she holds her tablet and, when the gripper is on its way to the shape (within 45 cm of the grasp), lays '
              'it over the hole (1 m/s) and keeps her hand on it (the tablet is held, not a free body) for 2.5 s, then '
              'withdraws it',
    }

    def build_scene(self) -> None:
        self.tablet = self.add_item('tablet', 'persons_tablet')
        # (set at load, before the simulator builds the sorter's bodies: it applies from then on)
        for s in self.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE):
            if s.is_respondable() and 'shape_sorter' in s.get_name():
                set_friction(s, SORTER_FRICTION)

    def setup_episode(self, rng: np.random.Generator) -> None:
        sorter = Shape('shape_sorter')
        # the top: highest respondable part of the sorter (its visual plate sticks out above the body)
        parts = [sorter] + [o for o in sorter.get_objects_in_tree(object_type=ObjectType.SHAPE) if o.is_respondable()]
        self.sorter_top = float(max(world_aabb(o)[1][2] for o in parts))
        m = sorter.get_matrix()
        sorter_yaw = float(np.arctan2(m[1, 0], m[0, 0]))
        self.sorter_yaw = sorter_yaw
        self.sorter_xy = np.asarray(sorter.get_position())[:2]
        centre = np.asarray(sorter.get_position())
        half = 0.5 * float(sorter.get_bounding_box()[1] - sorter.get_bounding_box()[0])
        self.sorter_half = half
        self.hole = np.asarray(Dummy('%s_drop_point' % SHAPE_NAMES[self.variation].replace(' ', '_')).get_position())
        self.shape = self.shapes[self.variation]
        self.grasp = np.asarray(self.grasp_points[self.variation].get_position())
        b = self.tablet
        R = np.array([[np.cos(sorter_yaw), -np.sin(sorter_yaw)], [np.sin(sorter_yaw), np.cos(sorter_yaw)]])
        local = R.T @ (self.hole[:2] - centre[:2])
        self.tablet_pos = None
        for turn in rng.permutation([0.0, np.pi / 2]):
            bh = b.half_extents[:2] if turn == 0.0 else b.half_extents[1::-1]
            slack = np.maximum(half - bh + 0.01, 0.0)
            c_local = np.clip(local + rng.uniform(-0.01, 0.01, 2), -slack, slack)
            if np.any(np.abs(local - c_local) > bh - 0.03):
                continue                                 # the hole would not be covered
            pos = np.r_[centre[:2] + R @ c_local, self.sorter_top + 0.002]
            if self.item_fits(b, pos, sorter_yaw + turn):
                self.tablet_pos, self.tablet_yaw = pos, float(sorter_yaw + turn)
                break
        if self.tablet_pos is None:
            raise BoundaryError('the tablet does not fit on the sorter')
        p = self.person
        drop = self.grip_palm(b, self.tablet_pos)
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng) or not p.arm_fits(drop):
            raise BoundaryError('the person cannot reach the sorter')
        self.reset_obstacle(b, 'tablet')
        self.parked = False
        self._held_prev = False
        if self.level == 'L0':
            b.place(self.tablet_pos, self.tablet_yaw)
        else:
            self.place_home(b, rng, self.tablet_yaw, reach=[drop])   # it lies in front of her, as it will on the sorter
            self.ready_in_hand(b)                                     # she has just picked it up

    def obstacle_in_way(self) -> bool:
        b = self.tablet
        if b.hidden or b.state != 'passive':
            return False
        f = b.frame()
        local = f[:3, :3].T @ (self.hole - f[:3, 3])
        return bool(np.all(np.abs(local[:2]) < b.half_extents[:2] - 0.005) and f[2, 3] > self.sorter_top - 0.02)

    def update(self, t: float) -> None:
        held = self.is_grasped(self.shape)
        if self.phase == 'waiting':
            if self.level == 'L1' and t >= 0.2:
                self.start_obstacle(self.job_move_item(self.tablet, self.tablet_pos, name='tablet_on_sorter',
                                                       speed=1.0))
            elif self.level == 'L2' and np.linalg.norm(self.tip_position() - self.grasp) < 0.45:
                # the robot is on its way to the shape: she lays her tablet over the hole and keeps her hand on it
                self.start_obstacle(self.job_move_item(self.tablet, self.tablet_pos, name='tablet_on_sorter',
                                                       speed=1.0, hold=2.5))
        elif self.phase == 'in_way' and self._held_prev and not held and not self.parked:
            self.parked = True
            self.log_event('shape_parked', at=[round(float(x), 3) for x in self.shape.get_position()])
        self._held_prev = held

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'tablet_pos'):
            return {}
        b = self.tablet
        return {'shape_parked': self.parked,
                'tablet_on_sorter': bool(not b.hidden and b.position()[2] > self.sorter_top - 0.03),
                'tablet_dropped': bool(not b.hidden and b.position()[2] < self.sorter_top - 0.05)}
