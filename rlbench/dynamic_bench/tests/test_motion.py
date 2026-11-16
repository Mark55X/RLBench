"""
Unit tests without the simulator: motion profiles and the oracle's trajectory scoring.
    python -m rlbench.dynamic_bench.tests.test_motion
"""

import numpy as np

from rlbench.dynamic_bench import motion, oracle
from rlbench.dynamic_bench.assets import matrix_to_quat, quat_to_matrix


def _num_speed(f, t0, t1, n=4000):
    ts = np.linspace(t0, t1, n)
    p = np.array([f(t) for t in ts])
    return np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(ts)


def test_profiles():
    for name, prof in motion.PROFILES.items():
        assert abs(prof(0.0)) < 1e-12 and abs(prof(1.0) - 1.0) < 1e-12, name
        assert prof(-1.0) == prof(0.0) and prof(2.0) == prof(1.0), name
        s = np.linspace(0, 1, 1001)
        v = np.diff([prof(x) for x in s]) / np.diff(s)
        assert np.all(v >= -1e-9), name                                  # monotone
        assert abs(v.max() - motion.PEAK_FACTOR[name]) < 0.01, (name, v.max())


def test_segment_peak_speed():
    for prof in ('cosine', 'min_jerk', 'linear'):
        seg = motion.Segment([0, 0, 0], [0.3, 0.4, 0], speed=0.1, profile=prof)
        v = _num_speed(seg.at, 0.0, seg.duration)
        assert abs(v.max() - 0.1) < 2e-3, (prof, v.max())               # 'speed' is the peak speed
        assert np.allclose(seg.at(seg.duration), [0.3, 0.4, 0]) and np.allclose(seg.at(0.0), 0)
        assert seg.done(seg.duration) and not seg.done(0.5 * seg.duration)


def test_chain_and_path():
    c = motion.path([[0, 0, 0], [1, 0, 0], [1, 1, 0]], speed=1.0, profile='linear', holds=[2.0])
    assert abs(c.duration - (1.0 + 2.0 + 1.0)) < 1e-9
    assert np.allclose(c.at(0.5), [0.5, 0, 0]) and np.allclose(c.at(2.0), [1, 0, 0])
    assert np.allclose(c.at(3.5), [1, 0.5, 0]) and np.allclose(c.at(100.0), [1, 1, 0])
    assert c.index(0.5) == 0 and c.index(2.0) == 1 and c.index(3.5) == 2 and c.index(10) == 3


def test_quaternions():
    q = motion.quat_from_axis_angle([0, 0, 1], np.pi / 2)
    assert np.allclose(q, motion.yaw_quat(np.pi / 2))
    assert np.allclose(motion.quat_mul(q, [0, 0, 0, 1]), q)
    assert np.allclose(motion.quat_mul(q, q), motion.yaw_quat(np.pi))
    rng = np.random.default_rng(0)
    for _ in range(50):                                                    # matrix <-> quaternion round trip
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = matrix_to_quat(quat_to_matrix(q))
        assert np.allclose(q2, q) or np.allclose(q2, -q)


def test_trajectory_scoring():
    rec = oracle.TrajectoryRecorder()
    for k in range(21):                                                    # a body moving at 0.1 m/s along x
        t = 0.05 * k
        rec.record(t, {'ball': ('passive', [0.1 * t, 0.0, 0.8], [0, 0, 0, 1])})
    gt = rec.ground_truth()['ball']
    assert gt['class'][0] == 'passive' and len(gt['t']) == 21
    assert np.allclose(oracle.position_at(gt, 0.525), [0.0525, 0.0, 0.8])
    assert oracle.position_at(gt, 2.0) is None
    perfect = oracle.prediction_error(gt, [0.2, 0.4, 0.6], [[0.02, 0, 0.8], [0.04, 0, 0.8], [0.06, 0, 0.8]])
    assert perfect['ade'] == 0.0 and perfect['n'] == 3
    still = oracle.prediction_error(gt, [0.5, 1.0, 3.0], [[0.0, 0, 0.8]] * 3)   # a constant-position guess
    assert still['n'] == 2 and abs(still['ade'] - 0.075) < 1e-6 and abs(still['fde'] - 0.1) < 1e-6


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print(f'{n} unit tests passed')
