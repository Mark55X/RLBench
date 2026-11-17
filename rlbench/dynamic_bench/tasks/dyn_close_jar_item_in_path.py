import numpy as np
from pyrep.objects.dummy import Dummy

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import TABLE_Z, DynamicBenchTask
from rlbench.tasks.close_jar import CloseJar

# tall kitchen items, all taller than the lid's way (the held lid's bottom passes 17 cm above the table). What the
# robot can do with them: the crackers box is too wide for the gripper (push only, low: pushed high it tips over),
# the mustard bottle can be grasped by its cap, the sugar box across its thin side (and tips over at a touch)
KINDS = ('crackers', 'mustard', 'sugar')
JAR_R = 0.05                 # [m] jar radius
LID_R = 0.038                # [m] lid radius
GAP = 0.06                   # [m] between the jar and the item in front of it (out of the gripper's turn on the jar)


class DynCloseJarItemInPath(DynamicBenchTask, CloseJar):
    """
    close_jar: the robot carries the lid from where it lies to above the target jar (static keyframes 'lid lifted'
    -> 'above the jar'). After the keyframes have been decided, the person (cooking at the same table) sets a tall
    kitchen item down right in front of the jar, on the side the lid comes from: the straight way between the two
    keyframes, at the keyframes' height, goes through it. The jar and the keyframes stay where they were.
    """

    BASE_TASK = 'close_jar'
    CATEGORY = 'obstacle on the way between two keyframes'
    OBSTACLE = 'a tall kitchen item (crackers box, mustard bottle or sugar box) set down in front of the jar'
    KEYFRAME_SEGMENT = 'lid lifted -> above the jar'
    SUMMARY = 'the person sets a tall kitchen item down in front of the jar, on the way of the lid'
    REMEDIES = ('carry the lid over the item (higher than the static way) or around it',
                'push the item aside low (pushed high it tips over: a side effect)',
                'park the lid, pick the item (mustard by its cap, sugar box across its thin side; the crackers box is '
                'too wide) and put it elsewhere, re-grasp the lid',
                'L2: the item arrives while the lid is on its way; let the soup can the person carries across pass')
    LEVEL_NOTES = {
        'L0': 'the item stands in front of the jar from the start',
        'L1': 'the person holds the item (taken from her side of the table) and, right after the start, sets it down '
              'in front of the jar (peak speed 1 m/s); it is at rest when the lid comes',
        'L2': 'as L1, and while the robot carries the lid the person carries a soup can across the lid\'s way and '
              'back (moving obstacles)',
    }

    def build_scene(self) -> None:
        self.kitchen = {k: self.add_item(k, 'kitchen_' + k) for k in KINDS}
        self.soup = self.add_item('soup', 'kitchen_soup')

    def setup_episode(self, rng: np.random.Generator) -> None:
        self.target_jar = self.jars[self.variation % 2]
        jar_xy = np.asarray(self.target_jar.get_position())[:2]
        self.w_lid = np.asarray(Dummy('waypoint2').get_position())       # lid lifted
        self.w_jar = np.asarray(Dummy('waypoint3').get_position())       # above the jar
        d = (self.w_lid - self.w_jar)[:2]
        d /= np.linalg.norm(d)
        p = self.person
        self.item_pos = None
        for kind in rng.permutation(KINDS):            # the first kitchen item that fits there and can be put there
            it = self.kitchen[str(kind)]
            # the wide side across the way
            base = float(np.arctan2(d[1], d[0]))
            yaw = base if it.half_extents[1] >= it.half_extents[0] else base + np.pi / 2
            c, s = np.cos(yaw - base), np.sin(yaw - base)
            ext = abs(c) * it.half_extents[0] + abs(s) * it.half_extents[1]          # half extent along d
            lid_xy = self.w_lid[:2]
            for extra in (0.0, 0.015, 0.03, 0.05, 0.08):
                xy = jar_xy + d * (JAR_R + GAP + ext + extra)
                if np.dot(lid_xy - xy, d) < LID_R + ext + 0.04:
                    break                                # anywhere on the way, but not where the lid is lifted
                pos = np.array([xy[0], xy[1], TABLE_Z + 0.001])
                if not self.item_fits(it, pos, yaw, gap=0.01):
                    continue
                drop = self.grip_palm(it, pos, 'pinch')
                if p.choose_seat([drop, drop + [0, 0, 0.10]], rng) and p.arm_fits(drop):
                    self.item_pos, self.item_yaw = pos, yaw
                    break
            if self.item_pos is not None:
                kind = str(kind)
                break
        if self.item_pos is None:
            raise BoundaryError('no kitchen item fits in front of the jar')
        self.reset_obstacle(it, kind)
        if self.level == 'L0':
            it.place(self.item_pos, self.item_yaw)
        else:
            home = self.place_home(it, rng, self.item_yaw, grip='pinch', reach=[drop])
            if self.level == 'L2':
                self.place_home(self.soup, rng, 0.0, grip='pinch', avoid=[(home, 0.16)], reach=[drop])
            self.ready_in_hand(it, 'pinch')            # she is about to put it down

    def obstacle_in_way(self) -> bool:
        """The item stands (its top above the lid's way) within reach of the lid's straight way."""
        it = self.obstacle
        if it is None or it.state != 'passive' or self.is_grasped(it.shape):
            return False
        a, b = self.w_lid[:2], self.w_jar[:2]
        q = it.position()[:2]
        u = np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.0, 1.0)
        near = np.linalg.norm(q - (a + u * (b - a))) < LID_R + 0.5 * it.radius + 0.01
        return bool(near and it.top_z() > self.w_jar[2] - 0.02)

    def update(self, t: float) -> None:
        if self.phase == 'waiting' and t >= 0.2:
            self.start_obstacle(self.job_move_item(self.obstacle, self.item_pos, name='item_in_front_of_jar',
                                                   speed=1.0, grip='pinch'))
        # L2: while the robot carries the lid, she carries a soup can across its way and back
        if self.level == 'L2' and self.phase != 'coming' and self.is_grasped(self.lid) and not self.person.busy:
            if self.start_traffic(self.soup, self.w_lid, self.w_jar, grip='pinch',
                                  stay_clear=[(self.w_jar, 0.12), (self.w_lid, 0.10),
                                              (self.item_pos, self.obstacle.radius + self.soup.radius + 0.04)]):
                self.finish_traffic(pause=0.5)

    def extra_metrics(self) -> dict:
        return {'lid_held_at_end': bool(self.is_grasped(self.lid))} if hasattr(self, 'item_pos') else {}
