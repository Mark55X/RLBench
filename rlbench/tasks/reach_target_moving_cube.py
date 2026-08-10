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

        # The cube that is moving
        self.cube = Shape('moving_cube')

    def init_episode(self, index: int) -> List[str]:
        # Constant velocity (10 cm/s)
        self.velocity = np.array([0.0, 0.1, 0.0])
        self.conditions = [
            TargetReachedCondition(self.target,
                                   self.success_sensor,
                                   self.robot.arm.get_tip(),
                                   distance_threshold=0.05)]
        self.register_success_conditions(self.conditions)

        return ['reach the red target while the cube moves',
                'touch the red sphere and ignore the moving block']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        current_pos = self.cube.get_position()
        new_pos = current_pos + self.velocity * 0.05
        self.cube.set_position(new_pos)

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True
