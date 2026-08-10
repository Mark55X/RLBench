from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

class ReachTargetMovingCube(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(),
                              ProximitySensor('success'))])

        # The cube that is moving
        self.cube = Shape('moving_cube')

    def init_episode(self, index: int) -> List[str]:
        # Constant velocity (10 cm/s)
        self.velocity = np.array([0.0, 0.1, 0.0])

        return ['reach the red target while the cube moves',
                'touch the red sphere and ignore the moving block']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        current_pos = self.cube.get_position()
        new_pos = current_pos + self.velocity * 0.05
        self.cube.set_position(new_pos)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True
