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
        # 1. Check Euclidean distance between tip dummy and target center
        tip_pos = np.asarray(self._tip.get_position())
        target_pos = np.asarray(self._target.get_position())
        distance = np.linalg.norm(tip_pos - target_pos)
        if distance <= self._distance_threshold:
            return True, False

        # 2. Check proximity sensor volume (using .read() to avoid V-REP -1 error with Dummy handles)
        if self._detector.still_exists():
            try:
                detected, _ = self._detector.read()
                if detected:
                    return True, False
            except Exception:
                pass

        return False, False


class DynamicDremaTest1(Task):
    """
    Dynamic Drema Test 1 Task:
    - Target: Red spherical marker on the tabletop (stays fixed across resets).
    - Dynamic Obstacle: Open-box / tunnel structure that oscillates symmetrically around
      the target along the Y axis.
    - Behavior: Starts from defined scene position, sweeps smoothly past the target to the
      symmetric opposite side of the table, and turns back symmetrically.
    """

    def __init__(self, pyrep, robot, name: str = 'dynamic_drema_test_1'):
        super().__init__(pyrep, robot, name=name)

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

        # Symmetric oscillation around the target:
        # Calculate signed offset from target Y to initial tunnel Y
        raw_offset = self.init_tunnel_pos[1] - self.target_pos[1]
        sign = -1.0 if raw_offset < 0 else 1.0
        # Ensure a wide excursion of at least 25 cm on each side of the target
        self.amplitude = max(abs(raw_offset), 0.25) * sign

        # Slower frequency (~350 steps per full cycle, ~85 steps to reach target from start)
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
        # Periodic symmetric oscillation around the target using cosine:
        # - step=0: cos(0) = 1 -> y = target_y + amplitude = init_tunnel_pos[1] (starts at scene pose)
        # - step=87: cos(pi/2) = 0 -> y = target_y (passes directly over target)
        # - step=175: cos(pi) = -1 -> y = target_y - amplitude (symmetric opposite side of target)
        # - then smoothly reverses and returns
        self.step_counter += 1
        delta_y = self.amplitude * np.cos(self.step_counter * self.oscillation_freq)
        new_y = self.target_pos[1] + delta_y

        # Keep safely within table lateral boundaries [-0.40, +0.40]
        new_y = float(np.clip(new_y, -0.40, 0.40))

        # Strictly preserve X and Z (table plane constraint)
        self.tunnel.set_position([self.init_tunnel_pos[0], new_y, self.init_tunnel_pos[2]])

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        # Must be True so RLBench does not randomly sample a new workspace/target position on reset!
        return True
