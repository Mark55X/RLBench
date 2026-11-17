import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import TABLE_Z, DynamicBenchTask
from rlbench.tasks.slide_block_to_color_target import SlideBlockToColorTarget

BLOCK_HALF = 0.029           # [m]
KINDS = {'mug': 'pinch', 'small_book': 'top'}


class DynSlideBlockObstacleInPath(DynamicBenchTask, SlideBlockToColorTarget):
    """
    slide_block_to_color_target: the robot pushes the block in a straight line from behind it (static keyframe
    'behind the block') to the coloured target (keyframe 'pushed'). After the keyframes have been decided, the person
    puts her mug or her book on the table in the block's way: pushed on, the block would push it along (a mug tips
    over). The keyframes stay valid.
    """

    BASE_TASK = 'slide_block_to_color_target'
    CATEGORY = 'obstacle on a pushing segment'
    OBSTACLE = 'the person\'s mug or book, in the block\'s way to the target'
    KEYFRAME_SEGMENT = 'behind the block -> pushed onto the target'
    SUMMARY = 'the person puts her mug or her book on the table in the way of the pushed block'
    REMEDIES = ('take the mug away (grasp it by the rim) or push the book aside, then push the block on',
                'push the block around the obstacle (a curved push ending at the same keyframe)',
                'pushing on regardless pushes the obstacle along (the mug tips over: a side effect)')
    LEVEL_NOTES = {
        'L0': 'the obstacle stands in the block\'s way from the start',
        'L1': 'the person holds it (just picked up from her side of the table) and, right after the start, puts it in '
              'the way (peak speed 1 m/s)',
        'L2': 'it comes when the gripper is about to push (within 12 cm of the keyframe behind the block, 1 m/s): it '
              'moves in while the block is being pushed',
    }

    ITEM_KINDS = KINDS                # what the person puts there (grip)
    # where along the block's way (fraction of its travel; the target counts the block on it at about half the
    # travel, so the obstacle stands in the first half)
    PLACES = (0.4, 0.3, 0.5)

    def build_scene(self) -> None:
        self.things = {k: self.add_item(k, 'persons_' + k) for k in self.ITEM_KINDS}

    def setup_episode(self, rng: np.random.Generator) -> None:
        w0 = np.asarray(Dummy('waypoint0').get_position())
        w1 = np.asarray(Dummy('waypoint1').get_position())
        self.c0 = np.asarray(self.block.get_position())
        self.w0 = w0
        u = (w1 - w0)[:2]
        self.push_len = float(np.linalg.norm(u))
        u /= self.push_len
        self.u = u
        # the block ends half a block (and a finger) ahead of the tip at the last keyframe
        self.end = w1[:2] + u * (BLOCK_HALF + 0.006)
        self.travel = float(np.linalg.norm(self.end - self.c0[:2]))
        self.target_xy = np.asarray(ProximitySensor('success%d' % (self.variation + 1)).get_position())[:2]
        kind = str(rng.choice(sorted(self.ITEM_KINDS)))
        it = self.things[kind]
        base = float(np.arctan2(u[1], u[0]))
        yaw = base if it.half_extents[1] >= it.half_extents[0] else base + np.pi / 2       # wide side across
        if kind == 'mug':
            yaw = float(rng.uniform(-np.pi, np.pi))
        c, s = np.cos(yaw - base), np.sin(yaw - base)
        ext = abs(c) * it.half_extents[0] + abs(s) * it.half_extents[1]
        self.pos = None
        spots = [self.c0[:2] + u * max(d * self.travel, BLOCK_HALF + 0.02 + ext) for d in self.PLACES]
        for xy in spots:
            pos = np.array([xy[0], xy[1], TABLE_Z + 0.001])
            if self.item_fits(it, pos, yaw, gap=0.01):
                self.pos, self.yaw = pos, yaw
                break
        if self.pos is None:
            raise BoundaryError(f'the {kind} does not fit in the way of the block')
        self.grip = self.ITEM_KINDS[kind]
        p = self.person
        drop = self.grip_palm(it, self.pos, self.grip)
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng) or not p.arm_fits(drop):
            raise BoundaryError('the person cannot reach the way of the block')
        self.reset_obstacle(it, kind)
        if self.level == 'L0':
            it.place(self.pos, self.yaw)
        else:
            self.place_home(it, rng, self.yaw, grip=self.grip, reach=[drop])
            self.ready_in_hand(it, self.grip)       # she has just picked it up

    def obstacle_in_way(self) -> bool:
        it = self.obstacle
        if it is None or it.state != 'passive' or self.is_grasped(it.shape):
            return False
        a, b = self.c0[:2], self.end
        q = it.position()[:2]
        s = np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.0, 1.0)
        return bool(np.linalg.norm(q - (a + s * (b - a))) < BLOCK_HALF + 0.6 * it.radius and
                    it.position()[2] > TABLE_Z - 0.02)

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if self.level == 'L1' and t >= 0.2:
            self.start_obstacle(self.job_move_item(self.obstacle, self.pos, name='obstacle_in_the_way', speed=1.0,
                                                   grip=self.grip))
        elif self.level == 'L2':
            # the gripper is coming down behind the block: the push is about to start
            if np.linalg.norm(self.tip_position() - self.w0) < 0.12:
                self.start_obstacle(self.job_move_item(self.obstacle, self.pos, name='obstacle_in_the_way',
                                                       speed=1.0, grip=self.grip))

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'pos'):
            return {}
        return {'block_pushed_m': round(float(np.linalg.norm(np.asarray(self.block.get_position())[:2] -
                                                              self.c0[:2])), 3)}

