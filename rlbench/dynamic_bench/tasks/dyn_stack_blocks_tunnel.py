import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.conditions import DetectedSeveralCondition, NothingGrasped
from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import ROBOT_BASE_XY, TABLE_Z, DynamicBenchTask, robot_reachable
from rlbench.tasks.stack_blocks import StackBlocks

SLIDE = 0.24                 # [m] the tunnel stands this far from the block, along its opening, before it is slid
INNER_HALF = (0.075, 0.055)
PUSH_FROM = (0.40, 0.64)     # [m] distance from the robot base where the gripper can start a low push comfortably


def pushable(xy) -> bool:
    d = float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2] - ROBOT_BASE_XY))
    return PUSH_FROM[0] <= d <= PUSH_FROM[1]


def push_ok(c, d) -> bool:
    """The roof push along d (from 11.5 cm behind the tunnel's centre c, 20 cm long) starts and ends where the
    gripper can work low."""
    c, d = np.asarray(c, dtype=np.float64)[:2], np.asarray(d, dtype=np.float64)[:2]
    return pushable(c - d * 0.115) and pushable(c + d * 0.085)  # [m] half length (opening axis) / half width of the tunnel's inside, with a margin


class DynStackBlocksTunnel(DynamicBenchTask, StackBlocks):
    """
    stack_blocks with a toy tunnel of the same game on the table (the tunnel of the first DynamicDrema tests). After
    the keyframes have been decided the person, tidying up, slides the tunnel along the table and it ends up right
    over a block the robot has to grasp: the block is still there and the keyframes are still right, but the
    top-down grasp hits the roof. Only the blocks to stack have the task's colour (the other cubes of the game take
    the distractors' colours): no other block can replace the covered one. The tunnel is light (0.3 kg) and a free body: it can be slid off along its opening
    (the block stays where it is) or lifted off by a wall.
    """

    BASE_TASK = 'stack_blocks'
    CATEGORY = 'keyframe target covered from above'
    OBSTACLE = 'a toy tunnel slid over a block the robot has to grasp'
    KEYFRAME_SEGMENT = 'above the block -> grasp'
    SUMMARY = 'the person slides a toy tunnel along the table and it ends up over a block the robot has to grasp'
    REMEDIES = ('slide the tunnel off along its opening, pushing its roof (the block inside stays put)',
                'lift it off by a wall and put it aside',
                'pushing it sideways drags the block along (a wall hits it)')
    LEVEL_NOTES = {
        'L0': 'the tunnel stands over a block the robot has to grasp from the start',
        'L1': 'right after the start the person slides the tunnel (it stands on the table next to a block the robot '
              'has to grasp) over that block (peak speed 1 m/s, slide 0.25 m/s)',
        'L2': 'when the robot grasps its first block, the person slides the tunnel over a block it will grasp later: '
              'it moves while the robot carries the first block, and covers the next grasp',
    }

    def build_scene(self) -> None:
        self.tunnel = self.add_item('tunnel', 'toy_tunnel')

    def init_episode(self, index: int):
        descriptions = super().init_episode(index)
        # only the blocks to stack have the colour (the others take the distractors' colours, all the cubes stay):
        # the covered block cannot be swapped for another one, the tunnel has to go
        spare = self.target_blocks[self.blocks_to_stack:]
        for k, b in enumerate(spare):
            b.set_color(self.distractors[k % len(self.distractors)].get_color())
        self.register_success_conditions([
            DetectedSeveralCondition(self.target_blocks[:self.blocks_to_stack], ProximitySensor('stack_blocks_success'),
                                     self.blocks_to_stack),
            NothingGrasped(self.robot.gripper)])
        return descriptions

    def setup_episode(self, rng: np.random.Generator) -> None:
        # the block it ends up over: one the robot still has to grasp (L2: not the first, it covers the next grasp
        # while the robot carries the first), within easy reach of the robot (it must get to one end of the tunnel)
        first = 1 if self.level == 'L2' else 0
        cands = [b for b in self.target_blocks[first:self.blocks_to_stack]
                 if 0.40 <= np.linalg.norm(np.asarray(b.get_position())[:2] - ROBOT_BASE_XY) <= 0.62]
        if not cands:
            raise BoundaryError('no block to cover within easy reach of the robot')
        self.covered = cands[0]
        self.covered_index = [b.get_handle() for b in self.target_blocks].index(self.covered.get_handle())
        c = np.asarray(self.covered.get_position())
        self.over = np.array([c[0], c[1], TABLE_Z + 0.001])
        tn = self.tunnel
        p = self.person
        top = TABLE_Z + tn.height
        others = [s for s in list(self.target_blocks) + list(self.distractors)
                  if s.get_handle() != self.covered.get_handle()]
        self.start = None
        for a in rng.permutation(np.linspace(0.0, 2 * np.pi, 12, endpoint=False)):
            u = np.array([np.cos(a), np.sin(a)])
            start = np.r_[c[:2] + u * SLIDE, TABLE_Z + 0.001]
            # nothing else in the strip the tunnel sweeps (16 cm wide), the covered block passes through its opening
            seg_clear = True
            for s in others:
                q = np.asarray(s.get_position())[:2]
                t = np.clip(np.dot(q - c[:2], u) / SLIDE, 0.0, 1.0)
                if np.linalg.norm(q - (c[:2] + t * SLIDE * u)) < 0.08 + 0.045:
                    seg_clear = False
                    break
            if not seg_clear:
                continue
            if not (self.item_fits(tn, self.over, a) and self.item_fits(tn, start, a, gap=0.01)):
                continue
            if not any(push_ok(c[:2], sgn * u) for sgn in (1.0, -1.0)):
                continue
            palms = [p.palm_above(top, start[:2]), p.palm_above(top, c[:2])]
            if not p.choose_seat(palms + [palms[0] + [0, 0, 0.08]], rng):
                continue
            if all(p.arm_fits(q) for q in palms):
                self.start, self.yaw = start, float(a)
                break
        if self.start is None:
            raise BoundaryError('no way to slide the tunnel over the block')
        self.reset_obstacle(tn, 'tunnel')
        tn.place(self.over if self.level == 'L0' else self.start, self.yaw)

    def obstacle_in_way(self) -> bool:
        tn = self.tunnel
        if tn.hidden or tn.state != 'passive' or self.is_grasped(tn.shape) or tn.tilt() > np.radians(20):
            return False
        f = tn.frame()
        local = f[:3, :3].T @ (np.asarray(self.covered.get_position()) - f[:3, 3])
        return bool(abs(local[0]) < INNER_HALF[0] + 0.03 and abs(local[1]) < INNER_HALF[1])

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        held = any(self.is_grasped(b) for b in self.target_blocks)
        if (self.level == 'L1' and t >= 0.2 and not held) or (self.level == 'L2' and held):
            self.start_obstacle(self.job_slide(self.tunnel, self.over[:2], name='tunnel_slid', speed=1.0,
                                               slide_speed=0.25))

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'start'):
            return {}
        return {'covered_block': self.covered.get_name(),
                'tunnel_upright': bool(self.tunnel.tilt() < np.radians(20))}
