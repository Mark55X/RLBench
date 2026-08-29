from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition, DetectedCondition


class TargetReachedCondition(Condition):
    def __init__(self, target, detector, tip, distance_threshold: float = 0.05):
        self._target = target
        self._detector = detector
        self._tip = tip
        self._distance_threshold = distance_threshold

    def condition_met(self):
        detector_hit = self._detector.is_detected(self._target)
        if detector_hit:
            return True, False

        tip_pos = np.asarray(self._tip.get_position())
        target_pos = np.asarray(self._target.get_position())
        distance = np.linalg.norm(tip_pos - target_pos)
        return distance <= self._distance_threshold, False


class ReachTargetMovingCube(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.success_sensor = ProximitySensor('success')

        # Primary moving cube
        self.cube = Shape('moving_cube')
        # Secondary moving object (parallelepiped / obstacle)
        self.cube_1 = Shape('moving_cube_1')

    def init_episode(self, index: int) -> List[str]:
        # Save initial positions
        self.init_pos = np.array(self.cube.get_position())
        self.init_pos_1 = np.array(self.cube_1.get_position())

        # Reset positions to table surface
        self.cube.set_position(self.init_pos)
        self.cube_1.set_position(self.init_pos_1)

        # Velocities
        self.velocity = np.array([0.0, 0.1, 0.0])
        self.velocity_1 = np.array([0.08, -0.04, 0.0])

        self.conditions = [
            TargetReachedCondition(self.target,
                                   self.success_sensor,
                                   self.robot.arm.get_tip(),
                                   distance_threshold=0.05)]
        self.register_success_conditions(self.conditions)

        return ['reach the red target while multiple obstacles move',
                'touch the red sphere and avoid the moving blocks']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        # Move primary cube on table plane (Z strictly locked)
        current_pos = self.cube.get_position()
        new_x = current_pos[0] + self.velocity[0] * 0.05
        new_y = current_pos[1] + self.velocity[1] * 0.05
        self.cube.set_position([new_x, new_y, self.init_pos[2]])

        # Move secondary object on table plane (Z strictly locked)
        current_pos_1 = self.cube_1.get_position()
        new_x_1 = current_pos_1[0] + self.velocity_1[0] * 0.05
        new_y_1 = current_pos_1[1] + self.velocity_1[1] * 0.05
        self.cube_1.set_position([new_x_1, new_y_1, self.init_pos_1[2]])

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True
