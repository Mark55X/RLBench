import numpy as np
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import ROBOT_BASE_XY, TABLE_Z, DynamicBenchTask, world_aabb
from rlbench.dynamic_bench.person import TABLE_HALF_Y, TABLE_X
from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES, PlaceShapeInShapeSorter

TOPPLE_FROM = (0.33, 0.66)   # [m] distance from the robot base where the gripper can push high on the box comfortably


class DynShapeSorterBoxOverShape(DynamicBenchTask, PlaceShapeInShapeSorter):
    """
    place_shape_in_shape_sorter: after the keyframes have been decided, the person turns the toy box of the game upside
    down over the shape the robot has to grasp (static keyframes 'above the shape' -> 'grasp'). The shape is still
    there and the keyframes are still right, but the grasp from above hits the box. The box is too wide for the
    gripper and closed on every side: pushed low or slid it drags the shape along inside it; it has to be TOPPLED
    (pushed high on one side, following its edge as it tips about the opposite bottom edge), the shape stays where it
    is.
    """

    BASE_TASK = 'place_shape_in_shape_sorter'
    CATEGORY = 'keyframe target covered (topple)'
    OBSTACLE = 'the toy box of the game, upside down over the shape to grasp'
    KEYFRAME_SEGMENT = 'above the shape -> grasp'
    MAY_TOPPLE = ('toy_box',)         # toppling it is the remedy, not a side effect
    SUMMARY = 'the person turns the toy box upside down over the shape the robot has to grasp'
    REMEDIES = ('topple the box: push it high on one side, the gripper following the edge as it tips about the '
                'opposite bottom edge (the shape stays where it is), then grasp as decided',
                'it is too wide to grasp, and slid or pushed low it drags the shape along inside it')
    LEVEL_NOTES = {
        'L0': 'the box stands upside down over the shape from the start',
        'L1': 'the person holds the box (just picked up from her side of the table) and, right after the start, puts '
              'it over the shape (peak speed 1 m/s)',
        'L2': 'she holds the box and puts it over the shape when the gripper is on its way to it (within 65 cm of '
              'the grasp: it has just set off, 1 m/s): it comes down while the robot approaches',
    }

    def build_scene(self) -> None:
        self.box = self.add_item('toy_box', 'toy_box')

    def setup_episode(self, rng: np.random.Generator) -> None:
        self.shape = self.shapes[self.variation]
        lo, hi = world_aabb(self.shape)
        self.shape_xy = 0.5 * (lo[:2] + hi[:2])
        self.grasp = np.asarray(self.grasp_points[self.variation].get_position())
        sorter = Shape('shape_sorter')
        parts = [sorter] + [o for o in sorter.get_objects_in_tree(object_type=ObjectType.SHAPE) if o.is_respondable()]
        boxes = [world_aabb(o) for o in parts]
        self.sorter_lo = np.min([b[0][:2] for b in boxes], axis=0)
        self.sorter_hi = np.max([b[1][:2] for b in boxes], axis=0)
        bx = self.box
        self.pos = None
        for _ in range(12):
            yaw = float(rng.uniform(-np.pi / 4, np.pi / 4))
            pos = np.r_[self.shape_xy + rng.uniform(-0.008, 0.008, 2), TABLE_Z + 0.001]
            if not any(self.topple_ok(pos, d) for d in self.box_axes(yaw)):
                continue
            # (clear of the rest by 8 mm: the sorter's top plate sticks out over the table at the box's height)
            if self.item_fits(bx, pos, yaw, gap=0.008, gap_ignore=[self.shape.get_handle()]):
                self.pos, self.yaw = pos, yaw
                break
        if self.pos is None:
            raise BoundaryError('the box does not fit over the shape (or could not be toppled)')
        p = self.person
        drop = self.grip_palm(bx, self.pos)
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng) or not p.arm_fits(drop):
            raise BoundaryError('the person cannot reach the shape')
        self.reset_obstacle(bx, 'box')
        if self.level == 'L0':
            bx.place(self.pos, self.yaw)
        else:
            self.place_home(bx, rng, self.yaw, reach=[drop])
            self.ready_in_hand(bx)                     # she has just picked it up

    # --- geometry ---------------------------------------------------------------------------------------------------

    @staticmethod
    def box_axes(yaw: float):
        return [np.array([np.cos(yaw + k * np.pi / 2), np.sin(yaw + k * np.pi / 2)]) for k in range(4)]

    def topple_ok(self, c, d) -> bool:
        """The box standing at c can be toppled along d: the gripper can push high on its side (behind it, along -d)
        and follow it, and it lies down on free table (not on the sorter, not off the table)."""
        c, d = np.asarray(c, dtype=np.float64)[:2], np.asarray(d, dtype=np.float64)[:2]
        hw = float(self.box.half_extents[0])
        start, pivot = c - d * (hw + 0.03), c + d * hw
        if not all(TOPPLE_FROM[0] <= np.linalg.norm(q - ROBOT_BASE_XY) <= TOPPLE_FROM[1] for q in (start, pivot)):
            return False
        n = np.array([-d[1], d[0]])
        lying = [pivot + d * a + n * b for a in (0.0, self.box.height) for b in (-hw, hw)]
        for q in lying:
            if not (TABLE_X[0] + 0.03 < q[0] < TABLE_X[1] - 0.03 and abs(q[1]) < TABLE_HALF_Y - 0.03):
                return False
        lo, hi = np.min(lying, axis=0), np.max(lying, axis=0)
        if np.all(lo < self.sorter_hi + 0.01) and np.all(hi > self.sorter_lo - 0.01):
            return False                                  # (bounding boxes: conservative)
        return True

    def obstacle_in_way(self) -> bool:
        bx = self.box
        if bx.hidden or bx.state != 'passive' or bx.tilt() > np.radians(30):
            return False
        f = bx.frame()
        local = f[:3, :3].T @ (np.r_[self.shape_xy, TABLE_Z + 0.01] - f[:3, 3])
        return bool(np.all(np.abs(local[:2]) < bx.half_extents[:2]))

    # --- script -----------------------------------------------------------------------------------------------------

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if (self.level == 'L1' and t >= 0.2) or (self.level == 'L2' and not self.is_grasped(self.shape) and
                                                 np.linalg.norm(self.tip_position() - self.grasp) < 0.65):
            self._follow_shape()
            self.start_obstacle(self.job_move_item(self.box, self.pos, name='box_over_shape', speed=1.0))

    def _follow_shape(self) -> None:
        """Over the shape where it is now (RLBench's shapes may roll a little after the layout)."""
        lo, hi = world_aabb(self.shape)
        xy = 0.5 * (lo[:2] + hi[:2])
        if np.linalg.norm(xy - self.shape_xy) < 0.005:
            return
        pos = np.r_[xy + (self.pos[:2] - self.shape_xy), self.pos[2]]
        if self.item_fits(self.box, pos, self.yaw, gap=0.008, robot=False, gap_ignore=[self.shape.get_handle()]):
            self.pos, self.shape_xy = pos, xy

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'pos'):
            return {}
        return {'shape': SHAPE_NAMES[self.variation], 'box_toppled': bool(self.box.tilt() > np.radians(60))}
