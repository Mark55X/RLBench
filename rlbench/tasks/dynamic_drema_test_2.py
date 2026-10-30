from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition


def _quat_multiply(a, b) -> np.ndarray:
    """Hamilton product of quaternions in [x, y, z, w] order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _quat_x_axis(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)])


class CubePlacedCondition(Condition):
    """The cube rests on the table within xy_threshold of the place goal and the gripper no longer holds it."""

    def __init__(self, cube, goal, gripper, xy_threshold: float = 0.04, z_threshold: float = 0.02):
        self._cube = cube
        self._goal = goal
        self._gripper = gripper
        self._xy_threshold = xy_threshold
        self._z_threshold = z_threshold

    def condition_met(self):
        cube = np.asarray(self._cube.get_position())
        goal = np.asarray(self._goal.get_position())
        placed = (np.linalg.norm(cube[:2] - goal[:2]) <= self._xy_threshold
                  and abs(cube[2] - goal[2]) <= self._z_threshold
                  and self._cube not in self._gripper.get_grasped_objects())
        return bool(placed), False


class DynamicDremaTest2(Task):
    """
    Pick and place with a moving tunnel (extension of dynamic_drema_test_1):
    - the tunnel starts on the right of the cube and sweeps over it while the arm descends to grasp;
    - it waits on the left side until the cube is lifted, then sweeps back to the right, crossing the path the
      arm takes to carry the cube to the place goal on the left of the table.
    """

    PREGRASP_HEIGHT = 0.12
    CARRY_HEIGHT = 0.10      # lift / pre-place height of the tip above the cube rest height: inside the tunnel height
    PLACE_DROP = 0.01        # the cube is released this much above its rest height
    LIFT_TRIGGER = 0.04      # cube rise [m] that starts the tunnel return

    def __init__(self, pyrep, robot, name: str = 'dynamic_drema_test_2'):
        super().__init__(pyrep, robot, name=name)

    def init_task(self) -> None:
        self.cube = Shape('pick_cube')
        self.place_goal = Dummy('place_goal')
        self.tunnel = Shape('tunnel_obstacle')
        self.register_graspable_objects([self.cube])

    def _grasp_quaternion(self) -> List[float]:
        """Top-down tip orientation with the fingers aligned to the cube faces (cube yaw modulo 90 deg)."""
        q_tip = np.asarray(self.top_down_tip_quat, dtype=np.float64)
        finger_axis = _quat_x_axis(q_tip)
        yaw_tip = np.arctan2(finger_axis[1], finger_axis[0])
        yaw_cube = self.cube.get_orientation()[2]
        delta = (yaw_cube - yaw_tip + np.pi / 4) % (np.pi / 2) - np.pi / 4
        q_z = np.array([0.0, 0.0, np.sin(delta / 2), np.cos(delta / 2)])
        q = _quat_multiply(q_z, q_tip)
        return list(q / np.linalg.norm(q))

    def get_keyframes(self) -> List[dict]:
        """
        Tip-frame keyframes (PerAct layout). The grasp keyframes follow the cube until it is grasped; the place
        keyframes are fixed on the goal.
        """
        quat = self._grasp_quaternion() if not self.cube_grasped else self.carry_quat
        cube = np.asarray(self.cube_rest_pos if self.cube_grasped else self.cube.get_position(), dtype=np.float64)
        goal = np.asarray(self.place_goal.get_position(), dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])

        def kf(name, pos, gripper_open, tol, funnel, ignore_collisions=None):
            d = {'name': name, 'pose': [float(x) for x in pos] + [float(x) for x in quat], 'frame': 'tip',
                 'gripper_open': gripper_open, 'position_tolerance': tol, 'approach_tolerance_deg': 10.0,
                 'rotation_tolerance_deg': 15.0, 'approach_funnel': funnel}
            if ignore_collisions is not None:
                d['ignore_collisions'] = ignore_collisions
            return d

        return [
            kf('pregrasp', cube + self.PREGRASP_HEIGHT * up, 1.0, 0.03, False),
            kf('grasp', cube, 0.0, 0.01, True, ignore_collisions=True),
            kf('lift', cube + self.CARRY_HEIGHT * up, 0.0, 0.03, False),
            kf('preplace', goal + self.CARRY_HEIGHT * up, 0.0, 0.03, False),
            kf('place', goal + self.PLACE_DROP * up, 1.0, 0.015, True, ignore_collisions=True),
            kf('retreat', goal + self.PREGRASP_HEIGHT * up, 1.0, 0.03, False),
        ]

    def init_episode(self, index: int) -> List[str]:
        self.top_down_tip_quat = list(self.robot.arm.get_tip().get_quaternion())
        self.cube_rest_pos = np.array(self.cube.get_position())
        self.cube_grasped = False
        self.carry_quat = self._grasp_quaternion()

        tunnel_pos = np.array(self.tunnel.get_position())
        self.tunnel_x, self.tunnel_z = float(tunnel_pos[0]), float(tunnel_pos[2])
        self.right_y = float(np.clip(self.cube_rest_pos[1] - 0.25, -0.40, 0.40))
        self.left_y = float(np.clip(self.cube_rest_pos[1] + 0.25, -0.40, 0.40))
        self.tunnel.set_position([self.tunnel_x, self.right_y, self.tunnel_z])
        # ~175 steps per sweep, as the single pass of dynamic_drema_test_1
        self.sweep_rate = 0.018
        self.phase = 'first_sweep'
        self.sweep_step = 0

        self.register_success_conditions([
            CubePlacedCondition(self.cube, self.place_goal, self.robot.gripper)])
        return [
            'pick up the cube and place it on the left of the table while avoiding the moving tunnel',
            'move the cube to the green square, dodging the tunnel on the way',
            'pick and place dynamic drema test 2'
        ]

    def variation_count(self) -> int:
        return 1

    def _sweep(self, from_y: float, to_y: float) -> bool:
        self.sweep_step += 1
        s = min(self.sweep_step * self.sweep_rate, np.pi)
        y = from_y + (to_y - from_y) * 0.5 * (1.0 - np.cos(s))
        self.tunnel.set_position([self.tunnel_x, y, self.tunnel_z])
        return s >= np.pi

    def step(self) -> None:
        if not self.cube_grasped and self.cube in self.robot.gripper.get_grasped_objects():
            self.cube_grasped = True
            self.carry_quat = self._grasp_quaternion()
        if self.phase == 'first_sweep':
            if self._sweep(self.right_y, self.left_y):
                self.phase = 'wait'
        elif self.phase == 'wait':
            lifted = self.cube.get_position()[2] - self.cube_rest_pos[2] >= self.LIFT_TRIGGER
            if self.cube_grasped and lifted:
                self.phase = 'return_sweep'
                self.sweep_step = 0
        elif self.phase == 'return_sweep':
            if self._sweep(self.left_y, self.right_y):
                self.phase = 'done'

    def cleanup(self) -> None:
        pass

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True
