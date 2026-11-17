import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import DynamicBenchTask, world_aabb
from rlbench.dynamic_bench.person import HAND_HALF_THICK, Hold, Job, Move
from rlbench.tasks.insert_onto_square_peg import InsertOntoSquarePeg

NEAR = 0.07                  # [m] the hand is in the way while the palm is this close to the target spoke's axis


class DynInsertPegHandInWay(DynamicBenchTask, InsertOntoSquarePeg):
    """
    insert_onto_square_peg: after the keyframes have been decided, the person puts her hand on the top of the target
    spoke (holding it, as if to steady the toy), right where the ring has to come down (static keyframes 'above the
    spoke' -> 'ring released on the spoke'). A hand is an agent: it cannot be pushed and must not be touched; it
    leaves by itself. The robot has to wait for it (and predict it at L2, where it fidgets and comes back).
    """

    BASE_TASK = 'insert_onto_square_peg'
    CATEGORY = 'the person\'s hand at a keyframe'
    OBSTACLE = 'the person\'s hand, resting on the top of the target spoke'
    KEYFRAME_SEGMENT = 'above the spoke -> ring released on the spoke'
    SUMMARY = 'the person rests her hand on the top of the target spoke'
    REMEDIES = ('wait until the hand has gone (it must not be touched), then insert as decided',
                'keep the ring away from the hand meanwhile (above it, or aside)',
                'L2: predict the fidgeting hand; it comes back once more')
    LEVEL_NOTES = {
        'L0': 'the hand rests on the top of the spoke from the start, still, and leaves after 14 s',
        'L1': 'right after the start the hand comes to rest on the top of the spoke (1 m/s) for 14 s, then leaves',
        'L2': 'the hand comes when the robot has grasped the ring (1 m/s), fidgets on the spoke for 8 s, leaves, and '
              'comes back for 4 s after a 2 s pause',
    }

    def setup_episode(self, rng: np.random.Generator) -> None:
        centre = np.asarray(self._success_centre.get_position())
        peg = min([Shape(f'pillar{k}') for k in range(3)],
                  key=lambda s: np.linalg.norm(np.asarray(s.get_position())[:2] - centre[:2]))
        self.peg = peg
        self.peg_xy = np.asarray(peg.get_position())[:2]
        self.peg_top = float(world_aabb(peg)[1][2])
        p = self.person
        rough = np.r_[self.peg_xy, self.peg_top + 0.03]
        if not p.choose_seat([rough, rough + [0, 0, 0.10]], rng):
            raise BoundaryError('the person cannot reach the peg board')
        # the palm on the spoke's top (the hand comes from her shoulder, the fingers over the spoke)
        self.palm = np.array([self.peg_xy[0], self.peg_xy[1], self.peg_top + HAND_HALF_THICK + 0.004])
        if p.reach_lean(self.palm) is None or not p.arm_fits(self.palm, ignore=[peg.get_handle()]):
            raise BoundaryError('the hand cannot rest on the spoke')
        self.reset_obstacle(None, 'hand')
        self.jobs = []
        if self.level == 'L0':
            p.set_palm(self.palm)
            p.palm_target = self.palm.copy()
            job = Job('hand_on_spoke', [Hold(14.0), Move(self.palm + [0.0, 0.0, 0.12], 0.5, direct=True,
                                                         wait_for_robot=False)])
            job.after_contact = 'withdraw'
            self.jobs.append(self.submit(job))

    def obstacle_in_way(self) -> bool:
        p = self.person
        if p.hidden:
            return False
        palm = p.palm_world()
        return bool(np.linalg.norm(palm[:2] - self.peg_xy) < NEAR and palm[2] < self.peg_top + 0.08)

    def track_obstacle(self, t: float) -> None:
        """The hand: in the way while it is near the spoke, cleared when it has gone (by itself)."""
        here = self.obstacle_in_way()
        if self.phase == 'coming' and here:
            self.phase, self.t_placed = 'in_way', t
            self.log_event('obstacle_placed', obstacle='hand', in_way=True)
        elif self.phase == 'in_way' and not here:
            self.phase, self.t_cleared = 'cleared', t
            self.log_event('obstacle_cleared', obstacle='hand', by='the person')
            if self.level == 'L2' and len(self.jobs) == 2 and self.jobs[1].status in ('pending', 'running'):
                self.phase = 'coming'                     # she comes back

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if self.level == 'L1' and t >= 0.2:
            self.jobs.append(self.submit(self.job_hand_rest(self.palm, 14.0, name='hand_on_spoke', speed=1.0)))
            self.phase = 'coming'
            self.log_event('obstacle_coming', obstacle='hand')
        elif self.level == 'L2' and self.is_grasped(self._square_ring):
            self.jobs.append(self.submit(self.job_hand_rest(self.palm, 8.0, name='hand_fidgets', speed=1.0,
                                                            sway=0.015)))
            again = self.job_hand_rest(self.palm, 4.0, name='hand_back', speed=1.0, sway=0.015)
            again.steps.insert(0, Hold(2.0))
            self.jobs.append(self.submit(again))
            self.phase = 'coming'
            self.log_event('obstacle_coming', obstacle='hand')

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'palm'):
            return {}
        return {'hand_jobs': [j.status for j in self.jobs], 'ring_held_at_end': bool(self.is_grasped(
            self._square_ring))}
