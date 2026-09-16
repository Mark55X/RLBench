from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition


class TargetReachedCondition(Condition):
    def __init__(self, target, detector, tip, distance_threshold: float = 0.06):
        self._target = target
        self._detector = detector
        self._tip = tip
        self._distance_threshold = distance_threshold

    def condition_met(self):
        # Check if proximity detector senses the robot arm tip
        detector_hit = self._detector.is_detected(self._tip)
        if detector_hit:
            return True, False

        # Fallback to Euclidean distance between tip and target center
        tip_pos = np.asarray(self._tip.get_position())
        target_pos = np.asarray(self._target.get_position())
        distance = np.linalg.norm(tip_pos - target_pos)
        return distance <= self._distance_threshold, False


class DynamicDremaTest1(Task):
    """
    Dynamic Drema Test 1 Task:
    - Target: Red spherical marker on the tabletop.
    - Dynamic Obstacle: Open-box / tunnel structure (ceiling + 2 side walls) that oscillates
      periodically back and forth along the Y axis directly over the target.
    - Behavior: When the tunnel is over the target, top-down access is occluded and physically blocked.
      When the tunnel shifts away, the target is clear and reachable.
    """

    def init_task(self) -> None:
        self.target = Shape('target')
        self.success_sensor = ProximitySensor('success')

        # Open-box / tunnel obstacle
        if Shape.exists('tunnel_obstacle'):
            self.tunnel = Shape('tunnel_obstacle')
        elif Shape.exists('moving_cube'):
            # Fallback if loaded from generic template
            self.tunnel = Shape('moving_cube')
        else:
            raise RuntimeError("Could not find 'tunnel_obstacle' in dynamic_drema_test_1 scene.")

    def init_episode(self, index: int) -> List[str]:
        # Target position on the table
        self.target_pos = np.array(self.target.get_position())

        # Start strictly from the position defined in the scene / .ttm
        self.init_tunnel_pos = np.array(self.tunnel.get_position())
        self.tunnel.set_position(self.init_tunnel_pos)

        # Extended travel parameters:
        # - oscillation_amplitude: 0.28m travel (+/- 28 cm across table)
        # - oscillation_freq: 0.018 rad/step (wider period ~350 steps, avoids turning back prematurely)
        self.oscillation_amplitude = 0.28
        self.oscillation_freq = 0.018
        self.step_counter = 0

        self.conditions = [
            TargetReachedCondition(
                self.target,
                self.success_sensor,
                self.robot.arm.get_tip(),
                distance_threshold=0.06
            )
        ]
        self.register_success_conditions(self.conditions)

        return [
            'reach the target on the table while avoiding the oscillating tunnel',
            'touch the target when the moving box leaves it accessible',
            'reach target dynamic drema test 1'
        ]

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        # Periodic oscillation starting from the defined initial scene position
        self.step_counter += 1
        delta_y = self.oscillation_amplitude * np.sin(self.step_counter * self.oscillation_freq)
        new_y = self.init_tunnel_pos[1] + delta_y

        # Keep safely within table lateral boundaries [-0.40, +0.40]
        new_y = float(np.clip(new_y, -0.40, 0.40))

        # Strictly preserve X and Z (table plane constraint)
        self.tunnel.set_position([self.init_tunnel_pos[0], new_y, self.init_tunnel_pos[2]])

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        # Dynamic workspace containing moving obstacles
        return False
