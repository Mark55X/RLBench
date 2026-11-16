import numpy as np
from pyrep.objects.dummy import Dummy

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import TABLE_Z, DynamicBenchTask
from rlbench.dynamic_bench.person import Hold
from rlbench.tasks.slide_block_to_color_target import SlideBlockToColorTarget

BLOCK_HALF = 0.029           # [m]
ACROSS = (0.16, 0.13)        # [m] the book slides from this far on her side of the block's way to this far beyond


class DynSlideBlockBookAcross(DynamicBenchTask, SlideBlockToColorTarget):
    """
    slide_block_to_color_target: the person's book lies on the table next to the way the robot pushes the block
    along (static keyframes 'behind the block' -> 'pushed onto the target'). After the keyframes have been decided she
    slides it across the table, through the block's way, at the robot's height: a moving obstacle on the table. At L1
    it stops in the way; at L2 it goes across and back while the block is being pushed.
    """

    BASE_TASK = 'slide_block_to_color_target'
    CATEGORY = 'moving obstacle across a pushing segment'
    OBSTACLE = 'the person\'s book, slid across the table through the block\'s way'
    KEYFRAME_SEGMENT = 'behind the block -> pushed onto the target'
    SUMMARY = 'the person slides her book across the table, through the way the block is pushed along'
    REMEDIES = ('predict the sliding book: hold the push, let it pass, push on',
                'L1 (it stops in the way): push it aside, or push the block around it',
                'pushing into it shoves the book along with the block')
    LEVEL_NOTES = {
        'L0': 'the book lies across the block\'s way from the start',
        'L1': 'right after the start the person slides the book from beside the way into it (0.25 m/s); it stops '
              'there',
        'L2': 'when the gripper is about to push (within 12 cm of the keyframe behind the block) she slides it across '
              'the way and, after 1 s, back (0.3 m/s): it crosses the way twice while the block is being pushed',
    }

    def build_scene(self) -> None:
        self.book = self.add_item('small_book', 'persons_book')

    def setup_episode(self, rng: np.random.Generator) -> None:
        w0 = np.asarray(Dummy('waypoint0').get_position())
        w1 = np.asarray(Dummy('waypoint1').get_position())
        self.w0 = w0
        self.c0 = np.asarray(self.block.get_position())
        u = (w1 - w0)[:2]
        self.push_len = float(np.linalg.norm(u))
        u /= self.push_len
        self.u = u
        self.end = w1[:2] + u * (BLOCK_HALF + 0.006)             # the block ends ahead of the last tip pose
        self.travel = float(np.linalg.norm(self.end - self.c0[:2]))
        n = np.array([-u[1], u[0]])
        b = self.book
        p = self.person
        # its narrow side along the way (it slides across with its long side first)
        yaw = float(np.arctan2(u[1], u[0])) + (0.0 if b.half_extents[0] <= b.half_extents[1] else np.pi / 2)
        ext_u = float(min(b.half_extents[:2]))
        # her side of the way first
        toward = np.sign(np.dot(n, np.array([0.0, p.side]))) or 1.0
        self.mid = None
        for s in (0.5, 0.45, 0.6):       # in the first half of the travel (the target counts it from about half)
            mid = self.c0[:2] + u * (s * self.travel)
            if np.linalg.norm(mid - self.c0[:2]) < BLOCK_HALF + ext_u + 0.03:
                continue
            for sign in (toward, -toward):
                a = mid + sign * n * ACROSS[0]
                bb = mid - sign * n * ACROSS[1]
                pts = [np.r_[q, TABLE_Z + 0.001] for q in (a, mid, bb)]
                if not all(self.item_fits(b, q, yaw, gap=0.01) for q in pts):
                    continue
                palms = [p.palm_above(TABLE_Z + b.height, q[:2]) for q in pts]
                if p.choose_seat(palms, rng) and all(p.arm_fits(q) for q in palms):
                    self.mid, self.a, self.b, self.yaw = pts[1], pts[0], pts[2], yaw
                    break
            if self.mid is not None:
                break
        if self.mid is None:
            raise BoundaryError('no room to slide the book across the block\'s way')
        self.reset_obstacle(b, 'book')
        b.place(self.mid if self.level == 'L0' else self.a, self.yaw)

    def obstacle_in_way(self) -> bool:
        b = self.book
        if b.hidden or b.state != 'passive':
            return False
        q = b.position()[:2]
        a, e = self.c0[:2], self.end
        s = np.clip(np.dot(q - a, e - a) / max(np.dot(e - a, e - a), 1e-9), 0.0, 1.0)
        n = np.array([-self.u[1], self.u[0]])
        f = b.frame()
        ext = abs(np.dot(f[:2, 0], n)) * b.half_extents[0] + abs(np.dot(f[:2, 1], n)) * b.half_extents[1]
        return bool(np.linalg.norm(q - (a + s * (e - a))) < BLOCK_HALF + ext)

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if self.level == 'L1' and t >= 0.2:
            self.start_obstacle(self.job_slide(self.book, self.mid[:2], name='book_slid_into_the_way', speed=1.0,
                                               slide_speed=0.25))
        elif self.level == 'L2' and np.linalg.norm(self.tip_position() - self.w0) < 0.12:
            self.start_obstacle(self.job_slide(self.book, self.b[:2], name='book_slid_across', speed=1.0,
                                               slide_speed=0.30))
            back = self.job_slide(self.book, self.a[:2], name='book_slid_back', speed=1.0, slide_speed=0.30)
            back.steps.insert(0, Hold(1.0))
            self.submit(back)

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'mid'):
            return {}
        return {'block_pushed_m': round(float(np.linalg.norm(np.asarray(self.block.get_position())[:2] -
                                                              self.c0[:2])), 3)}
