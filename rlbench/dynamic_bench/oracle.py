"""
Oracle information of the benchmark: what each body IS physically (its dynamics class) and what it actually DID
(ground-truth trajectories).

Dynamics classes (dynbench_object_info / dynbench_dynamics_class):
- 'static'  : fixed scene parts (table, cupboard, rack, sorter, the chair): never move;
- 'passive' : free rigid bodies under physics (gravity, friction, contacts): they rest, slide, tip over, roll; a
              simulator can predict them from their state;
- 'agent'   : the person's body parts: kinematic, driven from outside (not by physics: they do not fall, the robot
              cannot push them); their motion has to be predicted from observation;
- 'carried' : a passive object while the person holds it: it moves with the agent (not a free body) until let go;
- 'held'    : an object in the robot's gripper.

The classes are meant as an oracle input (in the real lab a perception module would have to infer them); the
trajectories are NOT given to the robot: the recorder logs them so that a predictor can be scored afterwards
(prediction_error).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

CLASSES = ('static', 'passive', 'agent', 'carried', 'held')


class TrajectoryRecorder:
    """Pose of every non-static body at every active simulation step (position [m], quaternion [x, y, z, w], class)."""

    def __init__(self, every: int = 1):
        self.every = max(1, int(every))
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.bodies: Dict[str, dict] = {}

    def record(self, t: float, poses: Dict[str, tuple]) -> None:
        """:param poses: name -> (class, position (3), quaternion (4))."""
        self.n += 1
        if (self.n - 1) % self.every:
            return
        for name, (cls, pos, quat) in poses.items():
            b = self.bodies.setdefault(name, {'t': [], 'pos': [], 'quat': [], 'class': []})
            b['t'].append(round(float(t), 4))
            b['pos'].append([round(float(x), 5) for x in pos])
            b['quat'].append([round(float(x), 5) for x in quat])
            b['class'].append(cls)

    def ground_truth(self, names: Optional[Sequence[str]] = None) -> Dict[str, dict]:
        keys = self.bodies.keys() if names is None else [n for n in names if n in self.bodies]
        return {k: {kk: list(v) for kk, v in self.bodies[k].items()} for k in keys}


def position_at(body: dict, t: float) -> Optional[np.ndarray]:
    """Recorded position of a body at time t (linear interpolation; None outside the recording)."""
    ts = np.asarray(body['t'], dtype=np.float64)
    if len(ts) == 0 or t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
        return None
    pos = np.asarray(body['pos'], dtype=np.float64)
    return np.array([np.interp(t, ts, pos[:, k]) for k in range(3)])


def prediction_error(body: dict, times: Sequence[float], predicted: Sequence[Sequence[float]]) -> dict:
    """Scores a predicted trajectory of one body against the ground truth.
    :param body: a ground-truth entry (dynbench_ground_truth()[name]).
    :param times: absolute episode times [s] of the predicted positions.
    :param predicted: predicted positions [m], one per time.
    :return: ade (mean displacement error), fde (error at the last time with ground truth), n (points scored)."""
    errs = []
    for t, p in zip(times, predicted):
        gt = position_at(body, float(t))
        if gt is not None:
            errs.append(float(np.linalg.norm(gt - np.asarray(p, dtype=np.float64))))
    if not errs:
        return {'ade': None, 'fde': None, 'n': 0}
    return {'ade': round(float(np.mean(errs)), 4), 'fde': round(errs[-1], 4), 'n': len(errs)}
