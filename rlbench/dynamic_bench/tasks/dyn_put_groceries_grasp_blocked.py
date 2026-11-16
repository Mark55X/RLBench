import numpy as np
from pyrep.objects.dummy import Dummy

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.assets import _mesh_world
from rlbench.dynamic_bench.core import TABLE_Z, DynamicBenchTask
from rlbench.tasks.put_groceries_in_cupboard import GROCERY_NAMES, PutGroceriesInCupboard

GAP = 0.006                  # [m] the person puts the item this close to the target


class DynPutGroceriesGraspBlocked(DynamicBenchTask, PutGroceriesInCupboard):
    """
    put_groceries_in_cupboard: the person is unpacking the same shopping. After the keyframes have been decided she
    puts another grocery item of the scene right against the target item (6 mm away), on the side where a finger of
    the static grasp keyframe goes down: the finger would land on it. When nothing tall enough fits there (a tall
    target, e.g. the crackers box: the fingers only come down along its top), she lays a flat item on the target's
    top instead. The keyframe stays where it is; the item has to go (or the target has to come out of the gap).
    """

    BASE_TASK = 'put_groceries_in_cupboard'
    CATEGORY = 'keyframe target crowded'
    OBSTACLE = ('another grocery item of the scene, put right against the target on the side of a finger (or, for a '
                'tall target, laid on its top)')
    KEYFRAME_SEGMENT = 'above the item -> grasp'
    SUMMARY = ('the person puts another grocery item against the target, where a finger of the grasp goes down (or '
               'on its top)')
    REMEDIES = ('push the item away (it slides; a tall box pushed high tips over), or push it off the target\'s top',
                'pick it and put it elsewhere (sugar box across its thin side, mustard by its cap; the crackers box '
                'and the coffee can are too wide)',
                'drag the target out of the gap from its free side, then grasp it as decided')
    LEVEL_NOTES = {
        'L0': 'an item stands against the target from the start',
        'L1': 'right after the start the person moves an item against the target (peak speed 1.1 m/s)',
        'L2': 'the item comes when the gripper is within 50 cm of the target (1 m/s): it moves in while the robot '
              'comes down; if a second item fits against the other finger side, it is put there right after the start',
    }

    def setup_episode(self, rng: np.random.Generator) -> None:
        self.target = self.groceries[self.variation]
        tgt = self.task_object(self.target)
        tgt.rewrap()
        self.tgt = tgt
        w1 = Dummy('waypoint1')
        self.w_grasp = np.asarray(w1.get_position())
        self.w_above = np.asarray(Dummy('waypoint0').get_position())
        fa = w1.get_matrix()[:3, 1].copy()            # the fingers close along the tip's y axis
        fa[2] = 0.0
        fa /= np.linalg.norm(fa)
        yaw_t = tgt.yaw()
        faces = [np.array([np.cos(yaw_t + k * np.pi / 2), np.sin(yaw_t + k * np.pi / 2), 0.0]) for k in range(4)]
        faces.sort(key=lambda n: -np.dot(n, fa))
        sides = [faces[0], faces[3]]                  # the two faces the fingers come down along
        # items tall enough to reach the finger tips at the grasp keyframe
        need = self.w_grasp[2] - 0.015
        p = self.person
        c = tgt.position()
        if not p.choose_seat([np.r_[c[:2], TABLE_Z + 0.25]], rng):
            raise BoundaryError('the person cannot reach the target item')
        others = [self.task_object(s) for s in self.groceries if s.get_handle() != self.target.get_handle()]
        for o in others:
            o.rewrap()
        reachable = [o for o in others if self._reachable(o)]
        reachable.sort(key=lambda o: np.linalg.norm(o.position()[:2] - p.to_world(p.S_rest)[:2]))
        others = [o for o in reachable if TABLE_Z + o.height > need]
        # one item against a finger side (required); at L2 a second one against the other side, if one fits
        self.blockers = []
        for k, side in enumerate(sides):
            for o in others:
                if any(o is b for b, _, _ in self.blockers):
                    continue
                pos = self._flush(o, side)
                if pos is not None:
                    self.blockers.append((o, pos, side))
                    break
            if self.blockers and (self.level != 'L2' or len(self.blockers) == 2):
                break
        self.mode = 'against'
        if not self.blockers:
            # nothing tall enough fits against it (a tall target): a flat item laid on its top blocks the grasp
            top = float(_mesh_world(self.target)[:, 2].max())
            for o in reachable:
                if o.height > 0.09 or min(o.half_extents[:2]) < 0.025:
                    continue
                pos = np.r_[tgt.position()[:2], top + 0.002]
                palm = self.grip_palm(o, pos, self._grip(o))
                if p.reach_lean(palm) is None or not p.arm_fits(palm, ignore=o.handles()):
                    continue
                if self.item_fits(o, pos, o.yaw()):
                    self.blockers.append((o, pos, None))
                    self.mode = 'on_top'
                    break
        if not self.blockers:
            raise BoundaryError('no grocery item fits against or on the target')
        self.reset_obstacle(self.blockers[0][0], self.blockers[0][0].name)
        self.second_job = None
        if self.level == 'L0':
            for o, pos, _ in self.blockers:
                o.place(pos, o.yaw())

    @staticmethod
    def _grip(o) -> str:
        return 'top'                # in the crowd of groceries the person takes things from the top

    def _reachable(self, o) -> bool:
        """The person can take o from where it stands (her arm clear of its neighbours)."""
        p = self.person
        palm = self.grip_palm(o, o.position(), self._grip(o))
        return (p.reach_lean(palm) is not None and p.reach_lean(palm + [0, 0, 0.10]) is not None and
                p.arm_fits(palm, ignore=o.handles()) and p.arm_fits(palm + [0, 0, 0.10], ignore=o.handles()))

    def _flush(self, o, side):
        """Pose of item o standing GAP from the target's side whose outward normal is side (o keeps its
        orientation: a hand does not turn what it carries). Extents from the meshes (the objects are turned)."""
        yaw = o.yaw()
        ext_t = float(np.max((_mesh_world(self.target)[:, :2] - self.tgt.position()[:2]) @ side[:2]))
        ext = float(np.max((_mesh_world(o.shape)[:, :2] - o.position()[:2]) @ -side[:2]))
        p = self.person
        for gap in (GAP, 2 * GAP):
            xy = self.tgt.position()[:2] + side[:2] * (ext_t + gap + ext)
            pos = np.array([xy[0], xy[1], TABLE_Z + 0.001])
            palm = self.grip_palm(o, pos, self._grip(o))
            if p.reach_lean(palm) is None or not p.arm_fits(palm, ignore=o.handles()):
                continue
            if self.item_fits(o, pos, yaw, gap=0.004, gap_ignore=[self.target.get_handle()]):
                return pos
        return None

    def obstacle_in_way(self) -> bool:
        if self.mode == 'on_top':
            o, pos, _ = self.blockers[0]
            return bool(o.state == 'passive' and not self.is_grasped(o.shape) and
                        np.linalg.norm(o.position()[:2] - self.tgt.position()[:2]) < 0.04 and
                        o.position()[2] > pos[2] - 0.02)
        for o, pos, _ in self.blockers:
            if o.state == 'passive' and not self.is_grasped(o.shape) and o.tilt() < np.radians(30) and \
                    np.linalg.norm(o.position()[:2] - pos[:2]) < 0.025:
                return True
        return False

    def update(self, t: float) -> None:
        if self.phase != 'waiting' or self.level not in ('L1', 'L2'):
            return
        o, pos, _ = self.blockers[0]
        if self.level == 'L1' and t >= 0.2:
            self.start_obstacle(self.job_move_item(o, pos, name='item_against_target', speed=1.1,
                                                   grip=self._grip(o)))
        elif self.level == 'L2':
            if len(self.blockers) > 1 and self.second_job is None and t >= 0.2:
                o2, pos2, _ = self.blockers[1]
                self.second_job = self.submit(self.job_move_item(o2, pos2, name='other_side_item', speed=0.9,
                                                                 grip=self._grip(o2)))
            if np.linalg.norm(self.tip_position() - self.w_grasp) < 0.50:
                self.start_obstacle(self.job_move_item(o, pos, name='item_against_target', speed=1.0,
                                                       grip=self._grip(o)))

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'blockers'):
            return {}
        return {'target': GROCERY_NAMES[self.variation], 'mode': self.mode,
                'blockers': [o.name for o, _, _ in self.blockers],
                'second_job': None if self.second_job is None else self.second_job.status,
                'target_grasped_at_end': bool(self.is_grasped(self.target))}
