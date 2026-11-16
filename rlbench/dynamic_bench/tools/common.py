"""Shared helpers of the benchmark tools: environment, snapshot camera, demo replay."""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


def make_env(headless: bool = True) -> Environment:
    oc = ObservationConfig()
    oc.set_all(False)
    oc.set_all_low_dim(True)
    env = Environment(MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()), obs_config=oc,
                      headless=headless, robot_setup='panda')
    env.launch()
    return env


class SnapshotCamera:
    """A free camera for pictures of the scene (not one of RLBench's observation cameras)."""

    def __init__(self, eye: Sequence[float] = (1.35, -0.85, 1.55), look_at: Sequence[float] = (0.25, 0.0, 0.85),
                 resolution: Tuple[int, int] = (480, 360), fov_deg: float = 50.0):
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode
        self.sensor = VisionSensor.create(list(resolution), view_angle=fov_deg, near_clipping_plane=0.05,
                                          far_clipping_plane=6.0, render_mode=RenderMode.OPENGL3,
                                          explicit_handling=True)
        self.look(eye, look_at)

    def look(self, eye: Sequence[float], look_at: Sequence[float]) -> None:
        eye, look_at = np.asarray(eye, float), np.asarray(look_at, float)
        z = look_at - eye
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
        self.sensor.set_matrix(m)

    def capture(self) -> np.ndarray:
        self.sensor.handle_explicitly()
        rgb = self.sensor.capture_rgb()
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def remove(self) -> None:
        self.sensor.remove()


def save_png(img: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(img).save(path)


def montage(images: List[np.ndarray], labels: List[str], cols: int = 3) -> np.ndarray:
    from PIL import Image, ImageDraw
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    for k, (img, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(img)
        ImageDraw.Draw(tile).text((8, 8), label, fill=(0, 0, 0))
        canvas.paste(tile, ((k % cols) * w, (k // cols) * h))
    return np.asarray(canvas)


from rlbench.dynamic_bench.keyframes import record_static_demo  # noqa: E402  (kept here for the tools)


def replay_open_loop(task_env, demo, max_extra_steps: int = 40, snapshot=None, snapshot_every: int = 0,
                     on_step=None, settle_steps: int = 4, tolerance: float = 0.01) -> Tuple[bool, int, List[np.ndarray]]:
    """
    Plays the joint positions of a static demo open loop (position targets, dynamics on): the 'plan once, execute
    blind' baseline. Each demo pose is held for up to settle_steps steps until every joint is within tolerance [rad]
    (the position controller lags behind the demo, which moved the joints kinematically). Returns (success, steps,
    snapshots).
    """
    shots = []
    success = False
    step = 0
    arm = task_env._task.robot.arm
    for obs in demo:
        action = np.concatenate([obs.joint_positions, [1.0 if obs.gripper_open > 0.5 else 0.0]])
        try:
            for _ in range(max(1, settle_steps)):
                task_env.step(action)
                if np.max(np.abs(np.asarray(arm.get_joint_positions()) - obs.joint_positions)) < tolerance:
                    break
        except Exception:
            break
        step += 1
        if on_step is not None:
            on_step(step)
        if snapshot is not None and snapshot_every and step % snapshot_every == 0:
            shots.append(snapshot.capture())
        success, _ = task_env._task.success()
        if success:
            break
    hold = np.concatenate([demo[-1].joint_positions, [1.0 if demo[-1].gripper_open > 0.5 else 0.0]])
    for _ in range(0 if success else max_extra_steps):
        task_env.step(hold)
        step += 1
        success, _ = task_env._task.success()
        if success:
            break
    return bool(success), step, shots
