import numpy as np
from pyrep.objects.dummy import Dummy

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import ROBOT_BASE_XY, TABLE_Z, DynamicBenchTask
from rlbench.dynamic_bench.person import TABLE_HALF_Y, TABLE_X
from rlbench.tasks.slide_block_to_color_target import SlideBlockToColorTarget

BLOCK_HALF = 0.029           # [m]
OVER = 0.07                  # [m] the board sticks out this far beyond the block's way (its free end)
SLIDE = 0.20                 # [m] at L1 / L2 the board lies this much further along the way (beyond the target) before
                             # she slides it in, towards the block
PUSH_FROM = (0.30, 0.68)     # [m] distance from the robot base where the gripper can push low comfortably


class DynSlideBlockBoardPinned(DynamicBenchTask, SlideBlockToColorTarget):
    """
    slide_block_to_color_target: the robot pushes the block in a straight line (static keyframes 'behind the block' ->
    'pushed onto the target'). After the keyframes have been decided the person lays a long wooden board across the
    table, across the block's way, and puts her book on one end. The board is too wide for the gripper
    and too flat to be taken by its edge; its far end is held down by the book, so it does not slide away as a whole:
    it has to be ROTATED about the end under the book (its free end swung along the way) until it lies beside the way.
    Pushing the block on regardless pushes the board and the book along.
    """

    BASE_TASK = 'slide_block_to_color_target'
    CATEGORY = 'obstacle on a pushing segment (rotate)'
    OBSTACLE = 'a long board across the block\'s way, its far end held down by the person\'s book'
    KEYFRAME_SEGMENT = 'behind the block -> pushed onto the target'
    SUMMARY = 'the person lays a long board across the block\'s way and puts her book on its far end'
    REMEDIES = ('rotate the board about its end under the book: push its free end along the way, the gripper '
                'following the arc, until it lies beside the way; then push the block on',
                'it is too wide to grasp; sliding it out lengthwise drags the book along; pushing the block on '
                'regardless pushes board and book along')
    LEVEL_NOTES = {
        'L0': 'the board lies across the way, the book on its far end, from the start',
        'L1': 'right after the start the person slides the board (lying beyond the target) along the way into it '
              '(0.25 m/s) and puts her book on its end',
        'L2': 'when the gripper is on its way behind the block (within 60 cm of the keyframe) she slides the board '
              'along the way towards the block (0.4 m/s), then puts her book on it: it arrives while the robot comes '
              'and pushes',
    }

    def build_scene(self) -> None:
        self.board = self.add_item('board', 'persons_board')
        self.book = self.add_item('book', 'persons_book')

    def setup_episode(self, rng: np.random.Generator) -> None:
        w0 = np.asarray(Dummy('waypoint0').get_position())
        w1 = np.asarray(Dummy('waypoint1').get_position())
        self.w0 = w0
        self.c0 = np.asarray(self.block.get_position())
        u = (w1 - w0)[:2]
        u /= np.linalg.norm(u)
        self.u = u
        self.end = w1[:2] + u * (BLOCK_HALF + 0.006)
        self.travel = float(np.linalg.norm(self.end - self.c0[:2]))
        n = np.array([-u[1], u[0]])
        bd, bk = self.board, self.book
        p = self.person
        half_l, half_w = float(bd.half_extents[0]), float(bd.half_extents[1])
        yaw = float(np.arctan2(n[1], n[0]))                      # the board's length across the way
        on_board = TABLE_Z + 0.001 + bd.height + 0.001
        toward = np.sign(np.dot(n, np.array([0.0, p.side]))) or 1.0
        self.mid = None
        why = {}
        for s in (0.5, 0.45, 0.6):
            # in the first half of the travel (the target counts the block on it from about half), the block 2 cm
            # behind it
            mid = self.c0[:2] + u * max(s * self.travel, BLOCK_HALF + half_w + 0.02)
            if np.linalg.norm(mid - self.c0[:2]) > 0.6 * self.travel + half_w:
                why['too far along the way'] = why.get('too far along the way', 0) + 1
                continue
            for sign in (toward, -toward):
                ax = sign * n                                     # from the way to the held (book) end
                centre = mid + ax * (half_l - OVER)
                pin = centre + ax * (half_l - bk.half_extents[0])     # the book flush with the far end
                home = centre + u * SLIDE
                pos = np.r_[centre, TABLE_Z + 0.001]
                if not all(TABLE_X[0] + 0.05 < q[0] < TABLE_X[1] - 0.05 and abs(q[1]) < TABLE_HALF_Y - 0.03
                           for q in (centre + ax * half_l, home + ax * half_l)):
                    why['off the table'] = why.get('off the table', 0) + 1
                    continue
                if not (self.item_fits(bd, pos, yaw, gap=0.01) and
                        self.item_fits(bd, np.r_[home, TABLE_Z + 0.001], yaw, gap=0.01)):
                    why['no room'] = why.get('no room', 0) + 1
                    continue
                swing = self.swing_direction(mid, ax, pin, why)
                if swing is None:
                    continue
                palms = [p.palm_above(TABLE_Z + bd.height, q) for q in (home, centre)] + \
                        [p.palm_above(on_board + bk.height, pin)]
                if p.choose_seat(palms, rng) and all(p.arm_fits(q) for q in palms):
                    self.mid, self.ax, self.yaw = mid, ax, yaw
                    self.board_pos, self.board_home = pos, np.r_[home, TABLE_Z + 0.001]
                    self.book_pos = np.r_[pin, on_board]
                    self.swing = swing
                    break
                why['person'] = why.get('person', 0) + 1
            if self.mid is not None:
                break
        if self.mid is None:
            raise BoundaryError(f'no room to lay the board across the block\'s way: {why}')
        self.reset_obstacle(bd, 'board')
        self.book_job = None
        if self.level == 'L0':
            bd.place(self.board_pos, self.yaw)
            bk.place(self.book_pos, self.yaw)
        else:
            bd.place(self.board_home, self.yaw)
            self.place_home(bk, rng, self.yaw, reach=[p.palm_above(on_board + bk.height, self.book_pos[:2])])

    def swing_direction(self, mid, ax, pin, why=None):
        """Along the way (+u or -u): the way the free end can be swung about the pin (the board's sweep keeps clear
        of the block, the gripper can follow the free end), or None."""
        half_l = float(self.board.half_extents[0])
        free = mid - ax * OVER                                        # the free end (on the board's axis)
        r = float(np.linalg.norm(free - pin))
        for e in (self.u, -self.u):
            # the block must not be within the swept disc sector on that side
            v = self.c0[:2] - pin
            if np.dot(v, e) > -0.02 and np.linalg.norm(v) < r + BLOCK_HALF + 0.02:
                if why is not None:
                    why['the block is in the sweep'] = why.get('the block is in the sweep', 0) + 1
                continue
            arc = [pin + (free - pin) * np.cos(a) + e * r * np.sin(a) for a in np.radians([0, 30, 60, 85])]
            if all(PUSH_FROM[0] <= np.linalg.norm(q - ROBOT_BASE_XY) <= PUSH_FROM[1] for q in arc) and \
                    all(TABLE_X[0] + 0.05 < q[0] < TABLE_X[1] - 0.05 and abs(q[1]) < TABLE_HALF_Y - 0.05 for q in arc):
                return e
            if why is not None:
                k = 'arc out of reach ' + str([round(float(np.linalg.norm(q - ROBOT_BASE_XY)), 2) for q in arc])
                why[k] = why.get(k, 0) + 1
        return None

    def obstacle_in_way(self) -> bool:
        bd = self.board
        if bd.hidden or bd.state != 'passive':
            return False
        f = bd.frame()
        a, e = self.c0[:2], self.end
        for t in np.linspace(0.0, 1.0, 25):
            q = a + t * (e - a)
            local = f[:3, :3].T @ (np.r_[q, f[2, 3]] - f[:3, 3])
            if abs(local[0]) < bd.half_extents[0] + BLOCK_HALF and abs(local[1]) < bd.half_extents[1] + BLOCK_HALF:
                return True
        return False

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if (self.level == 'L1' and t >= 0.2) or \
                (self.level == 'L2' and np.linalg.norm(self.tip_position() - self.w0) < 0.60):
            # (her hand on the board's end that the book will hold down, away from the way and the gripper)
            self.start_obstacle(self.job_slide(self.board, self.board_pos[:2], name='board_across_the_way', speed=1.0,
                                               slide_speed=0.25 if self.level == 'L1' else 0.40,
                                               grip_offset=self.ax * 0.10))
            self.book_job = self.submit(self.job_move_item(self.book, self.book_pos, name='book_on_the_board',
                                                           speed=1.0))

    def pivot(self) -> np.ndarray:
        """The point the board turns about: under the book if it is on the board, else its far end."""
        f = self.board.frame()
        axis = f[:2, 0]
        c = f[:2, 3]
        s = float(np.dot(self.book.position()[:2] - c, axis))
        if abs(s) < self.board.half_extents[0] and self.book.position()[2] > TABLE_Z + 0.5 * self.board.height:
            return c + axis * s
        far = c + axis * self.board.half_extents[0]
        near = c - axis * self.board.half_extents[0]
        return far if np.linalg.norm(far - self.mid) > np.linalg.norm(near - self.mid) else near

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'mid'):
            return {}
        return {'block_pushed_m': round(float(np.linalg.norm(np.asarray(self.block.get_position())[:2] -
                                                              self.c0[:2])), 3),
                'board_turned_deg': round(float(np.degrees(abs(np.arctan2(np.sin(self.board.yaw() - self.yaw),
                                                                          np.cos(self.board.yaw() - self.yaw))))), 1),
                'book_on_board': bool(self.book.position()[2] > TABLE_Z + 0.5 * self.board.height)}
