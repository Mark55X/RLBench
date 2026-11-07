from typing import List

import numpy as np

from rlbench.tasks.dynamic_drema_test_2 import DynamicDremaTest2


class DynamicDremaTest3(DynamicDremaTest2):
    """
    Pick and place with a tunnel that parks over the cube (variant of dynamic_drema_test_2, scenario 3a):
    - the tunnel starts on the right of the cube and sweeps over it at the start of the episode, before the arm
      gets close (a kinematic sweep into the arm would be a physically impossible interpenetration);
    - it stops centred over the cube and stays there, so the top-down grasp of the keyframes is blocked;
    - once parked it becomes a light dynamic body that a push can move (PARKED_MASS).
    Without moving the tunnel the task cannot be solved: it is the test case for blocked keyframes and for
    planning a deliberate push.
    """

    PARKED_MASS = 0.3  # [kg] light tunnel, movable by a push
    PARK_SWEEP_RATE = 0.1  # ~30 steps: parked before the arm reaches the pregrasp, so it never sweeps into the arm

    def __init__(self, pyrep, robot, name: str = 'dynamic_drema_test_3'):
        super().__init__(pyrep, robot, name=name)

    def init_episode(self, index: int) -> List[str]:
        super().init_episode(index)
        self.park_y = float(self.cube_rest_pos[1])
        self.sweep_rate = self.PARK_SWEEP_RATE
        self.tunnel.set_dynamic(False)
        return ['pick up the cube and place it on the left of the table; a tunnel parks over the cube',
                'move the cube to the green square although the tunnel stops on top of it',
                'pick and place dynamic drema test 3']

    def step(self) -> None:
        if not self.cube_grasped and self.cube in self.robot.gripper.get_grasped_objects():
            self.cube_grasped = True
            self.carry_quat = self._grasp_quaternion()
        if self.phase == 'first_sweep':
            if self._sweep(self.right_y, self.park_y):
                self.phase = 'parked'
                self.tunnel.set_mass(self.PARKED_MASS)
                self.tunnel.set_respondable(True)
                self.tunnel.set_dynamic(True)

    def cleanup(self) -> None:
        self.tunnel.set_dynamic(False)
