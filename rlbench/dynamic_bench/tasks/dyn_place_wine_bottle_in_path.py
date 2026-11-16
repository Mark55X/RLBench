import numpy as np
from pyrep.objects.dummy import Dummy

from rlbench.backend.exceptions import BoundaryError
from rlbench.dynamic_bench.core import TABLE_Z, DynamicBenchTask
from rlbench.tasks.place_wine_at_rack_location import PlaceWineAtRackLocation

# the tip waypoints of the last two keyframes (in front of the location, in the location), per variation
INSERT = {0: (3, 4), 1: (5, 6), 2: (7, 8)}
HELD_R = 0.035               # [m] radius of the bottle the robot carries (lying)
HELD_AHEAD = 0.20            # [m] the held bottle's body ahead of the tip (it is held by the neck)


class DynPlaceWineBottleInPath(DynamicBenchTask, PlaceWineAtRackLocation):
    """
    place_wine_at_rack_location: the robot brings its bottle in front of the rack location (static keyframe 'in front
    of the location', the bottle lying, held by the neck) and slides it in (keyframe 'in the location'). After the
    keyframes have been decided, the person stands her own bottle (green glass) on the table right in front of the
    rack, where the robot's bottle has to be at the first of the two keyframes and on its way in. Her bottle stands
    on its base: pushed high it tips over (it may roll), pushed low it slides; it can be grasped by the neck.
    """

    BASE_TASK = 'place_wine_at_rack_location'
    CATEGORY = 'obstacle at a keyframe and on the linear way after it'
    OBSTACLE = 'the person\'s wine bottle (green), standing in front of the rack location'
    KEYFRAME_SEGMENT = 'in front of the location -> in the location'
    SUMMARY = 'the person stands her own wine bottle in front of the rack location'
    REMEDIES = ('stand the own bottle up somewhere (lying on the table it would roll), take the person\'s bottle by '
                'the neck and stand it elsewhere, re-grasp the own bottle, slide it in',
                'push the person\'s bottle away low, at its base (pushed high it tips over and rolls)',
                'L2: the bottle arrives while the robot\'s bottle is on its way; let the glass the person carries '
                'across pass')
    LEVEL_NOTES = {
        'L0': 'the person\'s bottle stands in front of the rack location from the start',
        'L1': 'the person holds her bottle (taken from her side of the table) and, right after the start, stands it in '
              'front of the location (peak speed 1 m/s)',
        'L2': 'first the person carries a glass across the robot\'s way (back afterwards) and takes her bottle in her '
              'hand; when the robot grasps its own bottle she brings hers (1 m/s): it moves in while the robot\'s '
              'bottle is on its way',
    }

    def build_scene(self) -> None:
        self.bottle = self.add_item('wine_bottle', 'persons_wine_bottle')
        for v in self.bottle.visuals:
            v.set_color([0.12, 0.36, 0.16])            # green glass: not the robot's bottle
        self.glass = self.add_item('cup', 'persons_glass')

    def setup_episode(self, rng: np.random.Generator) -> None:
        i, j = INSERT[self.variation]
        self.w_front = np.asarray(Dummy(f'waypoint{i}').get_position())
        self.w_in = np.asarray(Dummy(f'waypoint{j}').get_position())
        self.w_grasp = np.asarray(Dummy('waypoint1').get_position())
        u = (self.w_in - self.w_front)[:2]
        length = float(np.linalg.norm(u))
        u /= length
        self.u = u
        b = self.bottle
        yaw = float(rng.uniform(-np.pi, np.pi))
        self.bottle_pos = None
        for s in (0.35, 0.25, 0.45, 0.15, 0.55):
            xy = self.w_front[:2] + u * s * length
            pos = np.array([xy[0], xy[1], TABLE_Z + 0.001])
            if self.item_fits(b, pos, yaw, gap=0.01):
                self.bottle_pos, self.bottle_yaw = pos, yaw
                break
        if self.bottle_pos is None:
            raise BoundaryError('the person\'s bottle does not fit in front of the rack')
        p = self.person
        drop = self.grip_palm(b, self.bottle_pos, 'pinch')
        if not p.choose_seat([drop, drop + [0, 0, 0.10]], rng) or not p.arm_fits(drop):
            raise BoundaryError('the person cannot reach the front of the rack')
        self.reset_obstacle(b, 'wine_bottle')
        self.z0 = self.wine_bottle.get_position()[2]
        if self.level == 'L0':
            b.place(self.bottle_pos, self.bottle_yaw)
        else:
            home = self.place_home(b, rng, self.bottle_yaw, grip='pinch', reach=[drop])
            if self.level == 'L2':
                self.place_home(self.glass, rng, 0.0, grip='pinch', avoid=[(home, 0.15)], reach=[drop])
            else:
                self.ready_in_hand(b, 'pinch')          # L1: she is about to put it down

    def obstacle_in_way(self) -> bool:
        """The person's bottle stands where the robot's bottle sweeps between the two keyframes."""
        b = self.bottle
        if b.hidden or b.state != 'passive' or self.is_grasped(b.shape):
            return False
        a = self.w_front[:2]
        e = self.w_in[:2] + self.u * HELD_AHEAD
        q = b.position()[:2]
        s = np.clip(np.dot(q - a, e - a) / max(np.dot(e - a, e - a), 1e-9), 0.0, 1.0)
        near = np.linalg.norm(q - (a + s * (e - a))) < HELD_R + 0.033 + 0.005
        return bool(near and b.top_z() > self.w_front[2] - HELD_R)

    def update(self, t: float) -> None:
        if self.phase != 'waiting':
            return
        if self.level == 'L1' and t >= 0.2:
            self.start_obstacle(self.job_move_item(self.bottle, self.bottle_pos, name='bottle_in_front_of_rack',
                                                   speed=1.0, grip='pinch'))
        elif self.level == 'L2':
            if t >= 0.2:
                self.start_traffic(self.glass, self.w_grasp, self.w_front, grip='pinch',
                                   stay_clear=[(self.w_grasp, 0.12), (self.bottle_pos, 0.12)])
                self.get_ready(self.bottle, 'pinch')
            if self.is_grasped(self.wine_bottle):
                self.start_obstacle(self.job_move_item(self.bottle, self.bottle_pos, name='bottle_in_front_of_rack',
                                                       speed=1.0, grip='pinch'))
                self.finish_traffic(pause=0.5)

    def extra_metrics(self) -> dict:
        if not hasattr(self, 'bottle_pos'):
            return {}
        return {'persons_bottle_standing': bool(self.bottle.tilt() < np.radians(30)),
                'robots_bottle_held_at_end': bool(self.is_grasped(self.wine_bottle))}
