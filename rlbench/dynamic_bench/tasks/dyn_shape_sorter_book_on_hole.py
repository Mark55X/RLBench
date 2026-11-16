import numpy as np
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import DynamicBenchTask, world_aabb
from rlbench.tasks.place_shape_in_shape_sorter import SHAPE_NAMES, PlaceShapeInShapeSorter


class DynShapeSorterBookOnHole(DynamicBenchTask, PlaceShapeInShapeSorter):
    """
    place_shape_in_shape_sorter: after the keyframes have been decided, the person lays her book flat on the sorter,
    over the target hole: the static release keyframe is now above the book. A book lying flat is too wide for the
    gripper: it has to be slid (pushed) off the hole, with the shape still in the hand or after parking it.
    """

    BASE_TASK = 'place_shape_in_shape_sorter'
    CATEGORY = 'keyframe target covered'
    OBSTACLE = 'the person\'s book, lying flat on the sorter over the target hole'
    KEYFRAME_SEGMENT = 'above the hole -> release'
    SUMMARY = 'the person lays her book on the sorter, over the target hole'
    REMEDIES = ('slide the book off the hole (non-prehensile push: it is too wide to grasp), with the shape in the '
                'hand or after parking the shape',
                'L2: wait until the person takes her hand off the book (pushing it then would push her hand)',
                'then insert the shape')
    LEVEL_NOTES = {
        'L0': 'the book lies on the sorter over the hole from the start',
        'L1': 'the person holds her book (just picked up from her side of the table) and, right after the start, lays '
              'it over the hole (peak speed 1 m/s)',
        'L2': 'the book is laid when the robot has grasped the shape (1 m/s), and the person keeps her '
              'hand on it for 4 s (the book is held meanwhile, not a free body)',
    }

    def build_scene(self) -> None:
        self.book = self.add_item('book', 'persons_book')

    def setup_episode(self, rng: np.random.Generator) -> None:
        sorter = Shape('shape_sorter')
        # the top: highest respondable part of the sorter (its visual plate sticks out above the body)
        parts = [sorter] + [o for o in sorter.get_objects_in_tree(object_type=ObjectType.SHAPE) if o.is_respondable()]
        self.sorter_top = float(max(world_aabb(o)[1][2] for o in parts))
        m = sorter.get_matrix()
        sorter_yaw = float(np.arctan2(m[1, 0], m[0, 0]))
        self.sorter_yaw = sorter_yaw
        centre = np.asarray(sorter.get_position())
        half = 0.5 * float(sorter.get_bounding_box()[1] - sorter.get_bounding_box()[0])
        self.hole = np.asarray(Dummy('%s_drop_point' % SHAPE_NAMES[self.variation].replace(' ', '_')).get_position())
        self.shape = self.shapes[self.variation]
        b = self.book
        R = np.array([[np.cos(sorter_yaw), -np.sin(sorter_yaw)], [np.sin(sorter_yaw), np.cos(sorter_yaw)]])
        local = R.T @ (self.hole[:2] - centre[:2])
        self.book_pos = None
        for turn in rng.permutation([0.0, np.pi / 2]):
            bh = b.half_extents[:2] if turn == 0.0 else b.half_extents[1::-1]
            slack = np.maximum(half - bh + 0.01, 0.0)
            c_local = np.clip(local + rng.uniform(-0.01, 0.01, 2), -slack, slack)
            if np.any(np.abs(local - c_local) > bh - 0.015):
                continue                                 # the hole would not be covered
            pos = np.r_[centre[:2] + R @ c_local, self.sorter_top + 0.002]
            if self.item_fits(b, pos, sorter_yaw + turn):
                self.book_pos, self.book_yaw = pos, float(sorter_yaw + turn)
                break
        if self.book_pos is None:
            raise BoundaryError('the book does not fit on the sorter')
        p = self.person
        drop = self.grip_palm(b, self.book_pos)
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng) or not p.arm_fits(drop):
            raise BoundaryError('the person cannot reach the sorter')
        self.reset_obstacle(b, 'book')
        self.parked = False
        self._held_prev = False
        if self.level == 'L0':
            b.place(self.book_pos, self.book_yaw)
        else:
            self.place_home(b, rng, self.book_yaw, reach=[drop])          # her book lies in front of her, as it will lie on the sorter
            self.ready_in_hand(b)                   # she has just picked it up

    def obstacle_in_way(self) -> bool:
        b = self.book
        if b.hidden or b.state != 'passive':
            return False
        f = b.frame()
        local = f[:3, :3].T @ (self.hole - f[:3, 3])
        return bool(np.all(np.abs(local[:2]) < b.half_extents[:2] - 0.005) and f[2, 3] > self.sorter_top - 0.02)

    def update(self, t: float) -> None:
        held = self.is_grasped(self.shape)
        if self.phase == 'waiting':
            if self.level == 'L1' and t >= 0.2:
                self.start_obstacle(self.job_move_item(self.book, self.book_pos, name='book_on_sorter', speed=1.0))
            elif self.level == 'L2' and held:
                self.start_obstacle(self.job_move_item(self.book, self.book_pos, name='book_on_sorter', speed=1.0,
                                                       hold=4.0))
        elif self.phase == 'in_way' and self._held_prev and not held and not self.parked:
            self.parked = True
            self.log_event('shape_parked', at=[round(float(x), 3) for x in self.shape.get_position()])
        self._held_prev = held

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'book_pos'):
            return {}
        b = self.book
        return {'shape_parked': self.parked,
                'book_on_sorter': bool(not b.hidden and b.position()[2] > self.sorter_top - 0.03)}
