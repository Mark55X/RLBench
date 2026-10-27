from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition


def _quat_z_axis(quat_xyzw) -> np.ndarray:
    """Third column of the rotation matrix of a unit quaternion [x, y, z, w]."""
    x, y, z, w = quat_xyzw
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


class TargetReachedCondition(Condition):
    """
    Reached when the gripper tip is within distance_threshold of the target center AND the gripper
    axis (tip z-axis) points down within max_approach_angle_deg, i.e. the target is reached from above.
    Panda_tip shares its z-axis with the flange, so no frame conversion is needed for the axis check.
    """

    def __init__(self, target, tip, distance_threshold: float = 0.03, max_approach_angle_deg: float = 15.0):
        self._target = target
        self._tip = tip
        self._distance_threshold = distance_threshold
        self._max_approach_angle = np.radians(max_approach_angle_deg)

    def condition_met(self):
        distance = np.linalg.norm(np.asarray(self._tip.get_position()) - np.asarray(self._target.get_position()))
        z_tip = _quat_z_axis(self._tip.get_quaternion())
        approach_angle = np.arccos(np.clip(-z_tip[2], -1.0, 1.0))
        return bool(distance <= self._distance_threshold and approach_angle <= self._max_approach_angle), False


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
        self.target = Shape('reach_marker')
        self.success_sensor = ProximitySensor('success')

        # =====================================================================
        # TARGET ORIENTATION SPECIFICATION:
        # - Mode A (Recommended, Position-Only): set target_ee_orientation = None
        #   Allows Franka to reach the sphere in its most kinematically natural,
        #   un-twisted configuration with 4mm precision and zero rotational tension.
        # - Mode B (Natural Franka top-down grasp): [0.92388, -0.38268, 0.0, 0.0]
        #   Enforces Z-axis pointing down to table with joint 7 at neutral 45 deg.
        # - Mode C (Rigid horizontal alignment): [1.0, 0.0, 0.0, 0.0]
        # =====================================================================
        self.target_ee_orientation = [0.92388, -0.38268, 0.0, 0.0]
        self.pregrasp_height = 0.12  # [m] pre-grasp keyframe distance above the target along the gripper axis

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
        Returns desired target pose:
        - 3-DoF [x, y, z] if target_ee_orientation is None (position-only goal)
        - 7-DoF [x, y, z, qx, qy, qz, qw] if target_ee_orientation is specified
        """
        pos = list(self.target.get_position())
        if self.target_ee_orientation is not None:
            return pos + list(self.target_ee_orientation)
        return pos

    def get_keyframes(self) -> List[dict]:
        """
        Ordered gripper keyframes, recomputed from the current target pose (so they follow a moving
        target): a pre-grasp pose pregrasp_height above the target along the gripper axis, then the
        target itself. Same structure a keyframe policy (e.g. PerAct) would provide:
        'pose' [x, y, z(, qx, qy, qz, qw)], position / approach-axis tolerances to advance, gripper state,
        and 'approach_funnel': the pre-grasp lies in free space, so only the final descent is kept in the cone.
        """
        final_pose = self.get_target_ee_pose()
        target_pos = np.asarray(final_pose[:3], dtype=np.float64)
        z_tool = _quat_z_axis(final_pose[3:7]) if len(final_pose) >= 7 else np.array([0.0, 0.0, -1.0])
        pregrasp_pos = target_pos - self.pregrasp_height * z_tool
        return [
            {'name': 'pregrasp', 'pose': list(pregrasp_pos) + list(final_pose[3:]),
             'position_tolerance': 0.03, 'approach_tolerance_deg': 10.0, 'gripper_open': 1.0,
             'approach_funnel': False},
            {'name': 'reach', 'pose': list(final_pose),
             'position_tolerance': 0.01, 'approach_tolerance_deg': 10.0, 'gripper_open': 1.0,
             'approach_funnel': True},
        ]

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

        # Lateral starting position on one side (starts where oscillation begins at step=0)
        self.lateral_start_y = float(np.clip(self.target_pos[1] + self.amplitude, -0.40, 0.40))
        self.tunnel.set_position([self.init_tunnel_pos[0], self.lateral_start_y, self.init_tunnel_pos[2]])

        # Slower frequency (~350 steps per full cycle, ~85 steps to reach target from start)
        self.oscillation_freq = 0.018
        self.step_counter = 0

        self.conditions = [
            TargetReachedCondition(
                self.target,
                self.robot.arm.get_tip(),
                distance_threshold=0.03,
                max_approach_angle_deg=15.0
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

        # --- MODE 1: COMPLETELY STATIC TUNNEL (Stationary at lateral side, away from target) ---
        # new_y = float(self.lateral_start_y)

        # --- MODE 2: SINGLE PASS (Sweeps across target once and halts at opposite side) ---
        phase = min(self.step_counter * self.oscillation_freq, np.pi)
        new_y = self.target_pos[1] + self.amplitude * np.cos(phase)

        # --- MODE 3: DOUBLE PASS (Full round trip, then halts back at lateral side) ---
        # phase = min(self.step_counter * self.oscillation_freq, 2.0 * np.pi)
        # new_y = self.target_pos[1] + self.amplitude * np.cos(phase)

        # --- MODE 4: INFINITE CONTINUOUS OSCILLATION (Standard dynamic benchmark) ---
        # new_y = self.target_pos[1] + self.amplitude * np.cos(self.step_counter * self.oscillation_freq)


        new_y = float(np.clip(new_y, -0.40, 0.40))
        self.tunnel.set_position([self.init_tunnel_pos[0], new_y, self.init_tunnel_pos[2]])

    def cleanup(self) -> None:
        self.conditions = []

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        # Must be True so RLBench does not randomly sample a new workspace/target position on reset!
        return True
