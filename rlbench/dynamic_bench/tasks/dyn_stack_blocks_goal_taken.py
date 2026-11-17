from typing import Optional

import numpy as np
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import ROBOT_BASE_XY, TABLE_Z, DynamicBenchTask, world_aabb
from rlbench.tasks.stack_blocks import StackBlocks

BLOCK = 0.05                 # [m] block edge


class DynStackBlocksGoalTaken(DynamicBenchTask, StackBlocks):
    """
    stack_blocks: the person plays with the same blocks. After the keyframes have been decided she puts one of the
    other blocks (another colour: a distractor of the task) on the stacking spot, the target of the static place
    keyframes. The block is a free body: it can be pushed off the spot (with the held block, or with the gripper
    after parking the held one) or picked and put elsewhere.
    L2 adds what the user's flagship scenario asks for: if the robot parks its block to clear the spot, the person
    stacks another distractor on the parked block (it has to be lifted off before the parked block can be taken).
    """

    BASE_TASK = 'stack_blocks'
    CATEGORY = 'keyframe target taken'
    OBSTACLE = 'a block of another colour (a distractor of the task) on the stacking spot'
    KEYFRAME_SEGMENT = 'place keyframes of the first block (above the target plane)'
    SUMMARY = 'the person moves a distractor block onto the stacking spot after the robot has started'
    REMEDIES = ('push the block off the spot (with the held block, or with the gripper after parking the held one)',
                'park the held block on a free spot, pick the distractor and put it away, re-grasp, stack',
                'L2: the block the person stacks on the parked one has to be lifted off first')
    LEVEL_NOTES = {
        'L0': 'a distractor block stands on the stacking spot from the start',
        'L1': 'right after the start the person moves a distractor block onto the stacking spot (peak speed 1.1 m/s);'
              ' it is at rest when the robot gets there',
        'L2': 'first the person carries a distractor across the robot\'s way and back; when '
              'the robot grasps its first block she puts another distractor onto the spot (1.1 m/s); if the robot '
              'parks its block, she stacks a distractor on the parked one',
    }

    def setup_episode(self, rng: np.random.Generator) -> None:
        plane = Shape('stack_blocks_target_plane')
        lo, hi = world_aabb(plane)
        self.plane_half = 0.5 * float(min(hi[0] - lo[0], hi[1] - lo[1]))
        c = 0.5 * (lo + hi)
        self.goal = np.array([c[0], c[1], hi[2]])
        self.on_goal = self.goal + [0.0, 0.0, 0.001]
        p = self.person
        drop = p.palm_above(self.goal[2] + BLOCK, self.goal[:2])
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng):
            raise BoundaryError('the person cannot reach the stacking spot')
        if not p.arm_fits(drop):
            raise BoundaryError('the person\'s arm would hit something at the stacking spot')
        shoulder = p.to_world(p.S_rest)
        cands = []
        for s in self.distractors:
            sp = s.get_position()
            palm = p.palm_above(sp[2] + 0.5 * BLOCK, sp[:2])
            if p.reach_lean(palm) is None or p.reach_lean(palm + [0, 0, 0.10]) is None:
                continue
            if np.linalg.norm(sp[:2] - self.goal[:2]) < 0.12 or not p.arm_fits(palm, ignore=[s.get_handle()]):
                continue
            cands.append(s)
        if not cands:
            raise BoundaryError('no distractor block within the person\'s reach')
        cands.sort(key=lambda s: np.linalg.norm(s.get_position()[:2] - shoulder[:2]))
        blk = self.task_object(cands[0])
        blk.rewrap()
        self.reset_obstacle(blk, 'block')
        self.traffic_blocks = []
        if self.level == 'L2':
            for s in cands[1:]:
                tb = self.task_object(s)
                tb.rewrap()
                self.traffic_blocks.append(tb)
        self.first = self.target_blocks[0]          # the static demo stacks the target blocks in order
        self.parked = None
        self.park_phase = 'none'
        self.second = None
        self.second_job = None
        self._held_prev = None
        if self.level == 'L0':
            if not self.item_fits(blk, self.on_goal, blk.yaw()):
                raise BoundaryError('the distractor does not fit on the stacking spot')
            blk.place(self.on_goal, blk.yaw())

    # --- state ------------------------------------------------------------------------------------------------------

    def _on_goal(self, pos, margin: float = 0.0) -> bool:
        return bool(np.all(np.abs(np.asarray(pos)[:2] - self.goal[:2]) < self.plane_half + margin))

    def obstacle_in_way(self) -> bool:
        b = self.obstacle
        if b is None or b.state != 'passive' or self.is_grasped(b.shape):
            return False
        return self._on_goal(b.position(), 0.5 * BLOCK) and b.position()[2] > self.goal[2] - 0.01

    def _held_target(self):
        for b in self.target_blocks:
            if self.is_grasped(b):
                return b
        return None

    # --- script -----------------------------------------------------------------------------------------------------

    def update(self, t: float) -> None:
        held = self._held_target()
        if self.phase == 'waiting':
            if self.level == 'L1' and t >= 0.2:
                self.start_obstacle(self.job_move_item(self.obstacle, self.on_goal, name='block_on_goal', speed=1.1))
            elif self.level == 'L2':
                if t >= 0.2:
                    if self.traffic_blocks and self.start_traffic(self.traffic_blocks, self.first.get_position(),
                                                                  self.goal, speed=0.9, stay_clear=[(self.goal, 0.13)]):
                        self.finish_traffic(0.5)        # and back: it may be the block she stacks on a parked one
                    else:
                        self._traffic_tried = 4
                if held is not None:
                    # the stacking spot is a 5 cm platform: the block is carried onto it (it cannot be slid up)
                    self.start_obstacle(self.job_move_item(self.obstacle, self.on_goal, name='block_on_goal',
                                                           speed=1.0))
        if self.level == 'L2':
            self._flagship(t, held)
        self._held_prev = held

    def _flagship(self, t: float, held) -> None:
        """If the robot parks its block while the spot is taken, the person stacks a distractor on it."""
        if self.park_phase == 'none':
            prev = self._held_prev
            if (prev is not None and held is None and self.phase == 'in_way' and
                    not self._on_goal(prev.get_position())):
                self.parked, self.parked_since = prev, t
                self.park_phase = 'parked'
                self.log_event('block_parked', block=prev.get_name(),
                               at=[round(float(x), 3) for x in prev.get_position()])
        elif self.park_phase == 'parked':
            pos = self.parked.get_position()
            settled = abs(pos[2] - (TABLE_Z + 0.5 * BLOCK)) < 0.01 and t - self.parked_since > 0.5
            if self.is_grasped(self.parked):
                self.park_phase = 'regrasped_first'
                self.log_event('parked_block_regrasped_before_second')
            elif settled and not self.tip_near(pos, 0.06) and not self.person.busy:
                choice = self._choose_second(self.parked)
                if choice is None and any(w in ('by the robot', 'by the gripper') for w in self._why_not.values()):
                    pass                                # a block is there once the robot has moved away: wait
                elif choice is None:
                    self.park_phase = 'second_impossible'
                    self.log_event('second_blocker_impossible', why=self._why_not)
                else:
                    self.second, on_top = choice
                    self.second_job = self.submit(self.job_move_item(self.second, on_top, name='stack_on_parked',
                                                                     speed=1.0))
                    self.park_phase = 'second_coming'
                    self.log_event('second_blocker_coming', block=self.second.name, on=self.parked.get_name())
        elif self.park_phase == 'second_coming':
            if self.second_job.status == 'done':
                self.park_phase = 'second_on'
                self.log_event('parked_block_covered', block=self.second.name)
            elif self.second_job.status == 'gave_up':
                self.park_phase = 'second_failed'
                self.log_event('second_blocker_failed', reason=self.second_job.reason)

    def _choose_second(self, parked) -> Optional[tuple]:
        """A block for the person to stack on the parked one: a distractor standing free within her reach (not the
        one on the spot, not by the robot), with the pose on top of the parked block."""
        top = world_aabb(parked)[1][2]
        pos = parked.get_position()
        on_top = np.array([pos[0], pos[1], top + 0.002])
        p = self.person
        self._why_not = {}
        if p.reach_lean(p.palm_above(top + BLOCK + 0.005, pos[:2])) is None:
            self._why_not['parked'] = 'out of her reach'
            return None
        cands = [s for s in self.distractors if s.get_handle() != self.obstacle.shape.get_handle()]
        cands.sort(key=lambda s: np.linalg.norm(s.get_position()[:2] - p.palm_world()[:2]))
        for s in cands:
            sp = s.get_position()
            why = self._why_not.__setitem__
            if self._on_goal(sp, 0.03) or abs(sp[2] - (TABLE_Z + 0.5 * BLOCK)) > 0.01 or self.is_grasped(s):
                why(s.get_name(), 'not standing free')
                continue
            if p.reach_lean(p.palm_above(sp[2] + 0.5 * BLOCK, sp[:2])) is None:
                why(s.get_name(), 'out of her reach')
                continue
            if np.linalg.norm(sp[:2] - self.tip_position()[:2]) < 0.15:
                why(s.get_name(), 'by the gripper')
                continue
            # not by the robot's body either (e.g. the block she carried across, left on the robot's side): she would
            # have to reach in front of the robot and wait for it
            if np.linalg.norm(sp[:2] - ROBOT_BASE_XY) < 0.40 or any(
                    np.linalg.norm(sp[:2] - np.asarray(j.get_position())[:2]) < 0.15 for j in self.robot.arm.joints):
                why(s.get_name(), 'by the robot')
                continue
            obj = self.task_object(s)
            obj.rewrap()
            return obj, on_top
        return None

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'park_phase'):
            return {}
        second_on = None
        if self.second is not None and self.parked is not None:
            d = self.second.position()[:2] - self.parked.get_position()[:2]
            second_on = bool(np.linalg.norm(d) < 0.02 and
                             self.second.position()[2] > self.parked.get_position()[2] + 0.02)
        return {'block_parked': self.parked is not None, 'park_phase': self.park_phase,
                'second_blocker_still_on_parked': second_on}
