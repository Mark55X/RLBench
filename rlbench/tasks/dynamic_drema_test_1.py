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
        # 1. Strictly check Euclidean distance between gripper tip and target center
        tip_pos = np.asarray(self._tip.get_position())
        target_pos = np.asarray(self._target.get_position())
        distance = np.linalg.norm(tip_pos - target_pos)
        if distance <= self._distance_threshold:
            return True, False

        # 2. Check proximity sensor volume only if tip is in close vicinity
        # (prevents oscillating tunnel passing over the target from triggering false success)
        if distance <= (self._distance_threshold * 1.5) and self._detector.still_exists():
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

        # Desired end-effector orientation for table interaction:
        # Franka Panda top-down approach (tool Z-axis pointing downward to tabletop)
        self.target_ee_orientation = [1.0, 0.0, 0.0, 0.0]

        # Open-box / tunnel obstacle
        if Shape.exists('tunnel_obstacle'):
            self.tunnel = Shape('tunnel_obstacle')
        elif Shape.exists('moving_cube'):
            # Fallback if loaded from generic template
            self.tunnel = Shape('moving_cube')
        else:
            raise RuntimeError("Could not find 'tunnel_obstacle' in dynamic_drema_test_1 scene.")

    def get_target_ee_pose(self) -> List[float]:
        """
        Returns desired 6-DoF end-effector pose [x, y, z, qx, qy, qz, qw] for reaching the target.
        """
        return list(self.target.get_position()) + list(self.target_ee_orientation)

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
        self.step_counter += 1

        # =====================================================================
        # SELECT TUNNEL MOTION MODE (Comment / Uncomment desired mode)
        # =====================================================================

        # --- MODE 1: COMPLETELY STATIC TUNNEL (Stays at initial scene position) ---
        new_y = float(self.init_tunnel_pos[1])

        # --- MODE 2: SINGLE PASS (Sweeps across target once and halts at opposite side) ---
        # phase = min(self.step_counter * self.oscillation_freq, np.pi)
        # new_y = self.target_pos[1] + self.amplitude * np.cos(phase)

        # --- MODE 3: DOUBLE PASS (Full round trip, then halts at initial side) ---
        # phase = min(self.step_counter * self.oscillation_freq, 2.0 * np.pi)
        # new_y = self.target_pos[1] + self.amplitude * np.cos(phase)

        # --- MODE 4: INFINITE CONTINUOUS OSCILLATION (Standard dynamic benchmark) ---
        # new_y = self.target_pos[1] + self.amplitude * np.cos(self.step_counter * self.oscillation_freq)

        # =====================================================================
        # Apply tunnel position (constrained to physical table boundaries)
        # =====================================================================
        new_y = float(np.clip(new_y, -0.40, 0.40))
        self.tunnel.set_position([self.init_tunnel_pos[0], new_y, self.init_tunnel_pos[2]])

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        # Must be True so RLBench does not randomly sample a new workspace/target position on reset!
        return True
