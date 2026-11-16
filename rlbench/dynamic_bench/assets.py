"""
Everyday objects of the benchmark, from the library that tools/build_assets.py extracts from RLBench's own task models
(rlbench/dynamic_bench/assets/*.ttm).

An Item is a passive rigid body modelled as RLBench models its task objects: a respondable dynamic shape (a simple
convex collision shape) with the realistic visual mesh as its child. Its frame (the pose given to set_pose) has the
origin at the centre of the footprint on the bottom face and z up when the object stands as it normally does (a bottle
on its base, a book lying flat, the cover cup upside down): set_pose([x, y, TABLE_Z], yaw_quat(a)) stands it on the
table.

Dynamics class of an item (what the oracle reports, see oracle.py):
- 'passive': a free rigid body under physics (gravity, contacts): it rests, slides, tips, rolls;
- 'carried': held by the person, kinematic, moved with the hand (not a free body until released);
- hidden items are not in the scene.
"""

import os
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from pyrep.backend import sim
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.objects.shape import Shape

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
PREFIX = 'dynbench_'
PARK_POSITION = np.array([3.0, -3.0, -2.0])     # far from the scene and below the floor: a hidden item

# kind: file in ASSET_DIR, realistic mass [kg] (RLBench's masses are off for some objects, e.g. a 3.8 kg tomato),
# euler [rad] turning the model into its standing pose, hollow (collision shape rebuilt as an open shell), collision
# ('cylinder': a 24-sided convex prism, 'box': a pure box, fitted to the model, replace RLBench's convex mesh: a
# can's convex mesh has a slightly rounded base that makes it rock and walk on a flat table in Bullet, and Bullet's
# pure cylinder jitters; a flat base stands still), damping
# (Bullet linear, angular). Every item gets a little angular damping (DAMPING): a standing can in Bullet otherwise
# starts rocking by itself now and then (about 1 in 16 placements) until it falls.
SPECS: Dict[str, dict] = {
    'mug': dict(file='mug', mass=0.30),
    'cup': dict(file='cup', mass=0.05),
    'cover_cup': dict(file='cup', mass=0.08, euler=(np.pi, 0.0, 0.0), hollow=True),   # upside down, open below
    'wine_bottle': dict(file='wine_bottle', mass=0.55, euler=(np.pi, 0.0, 0.0)),
    'soup': dict(file='soup', mass=0.35, collision='cylinder'),
    'tuna': dict(file='tuna', mass=0.17, collision='cylinder'),
    'spam': dict(file='spam', mass=0.34, collision='box'),
    'sugar': dict(file='sugar', mass=0.45, collision='box'),
    'coffee': dict(file='coffee', mass=0.60, collision='cylinder'),
    'mustard': dict(file='mustard', mass=0.40),
    'crackers': dict(file='crackers', mass=0.45, collision='box'),
    'chocolate_jello': dict(file='chocolate_jello', mass=0.12, collision='box'),
    'strawberry_jello': dict(file='strawberry_jello', mass=0.10, collision='box'),
    'book': dict(file='book', mass=0.35, collision='box'),
    'small_book': dict(file='small_book', mass=0.20, collision='box'),
    # the toy tunnel of the first DynamicDrema tests (lying on its side in its source model): roof up, open along x
    'tunnel': dict(file='tunnel', mass=0.30, euler=(0.0, -np.pi / 2, 0.0)),
}

DAMPING = (0.0, 0.3)



def _euler_matrix(euler: Sequence[float]) -> np.ndarray:
    """CoppeliaSim euler angles (alpha, beta, gamma: R = Rx(a) Ry(b) Rz(g))."""
    a, b, g = euler
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return rx @ ry @ rz


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x, y, z, w]."""
    m = np.asarray(m, dtype=np.float64)[:3, :3]
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _mesh_world(shape: Shape) -> np.ndarray:
    v, _, _ = shape.get_mesh_data()
    v = np.c_[np.asarray(v, dtype=np.float64).reshape(-1, 3), np.ones(len(v))]
    return (v @ shape.get_matrix().T)[:, :3]


class Item:
    """A passive everyday object of the scene (see the module docstring)."""

    def __init__(self, pyrep, kind: str, name: str, mass: Optional[float] = None):
        if kind not in SPECS:
            raise KeyError(f"Unknown item kind '{kind}' (one of {sorted(SPECS)})")
        spec = SPECS[kind]
        self.kind = kind
        self.name = PREFIX + name
        remove_named(self.name)
        for k in range(12):
            remove_named(f'{self.name}_visual{k if k else ""}')
        path = os.path.join(ASSET_DIR, spec['file'] + '.ttm')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} missing: run python -m rlbench.dynamic_bench.tools.build_assets')
        self.shape = Shape(pyrep.import_model(path).get_handle())
        self.shape.set_model(False)
        self.shape.set_dynamic(False)
        # standing pose of the model at the world origin
        m0 = np.eye(4)
        m0[:3, :3] = _euler_matrix(spec.get('euler', (0.0, 0.0, 0.0)))
        self.shape.set_matrix(m0)
        if spec.get('hollow'):
            self._make_hollow()
        elif spec.get('collision'):
            self._make_pure(spec['collision'])
        verts = _mesh_world(self.shape)
        lo, hi = verts.min(0), verts.max(0)
        centre = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
        # shape frame in the item frame (origin: footprint centre on the bottom face); the shape is in its standing
        # pose around the world origin now (the hollow shell has a frame of its own)
        self._local = self.shape.get_matrix()
        self._local[:3, 3] -= centre
        self.half_extents = 0.5 * (hi - lo)              # in the item frame (standing)
        self.height = float(hi[2] - lo[2])
        self.radius = float(np.linalg.norm(self.half_extents[:2]))     # footprint circumscribed radius
        self.visuals = [s for s in self.shape.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        self._self_visual = not self.visuals             # a plain shape is its own mesh (the tunnel)
        self.shape.set_name(self.name)
        for k, v in enumerate(self.visuals):
            v.set_name(f'{self.name}_visual{k if k else ""}')
        self.mass = float(spec['mass'] if mass is None else mass)
        # inertia from the collision shape (RLBench's models keep the inertia of their own masses), then the mass
        self.shape.compute_mass_and_inertia(1000.0)
        self.shape.set_mass(self.mass)
        for prop in (self.shape.set_collidable, self.shape.set_measurable, self.shape.set_detectable):
            prop(True)
        damping = spec.get('damping', DAMPING)
        h = self.shape.get_handle()
        sim.simSetEngineFloatParameter(sim.sim_bullet_body_lineardamping, h, float(damping[0]))
        sim.simSetEngineFloatParameter(sim.sim_bullet_body_angulardamping, h, float(damping[1]))
        self.state = 'hidden'
        self.hide()

    # --- collision shapes -------------------------------------------------------------------------------------------

    def _make_pure(self, kind: str) -> None:
        """Replaces the convex collision mesh with a pure cylinder / box fitted to it (standing pose, axis z)."""
        verts = _mesh_world(self.shape)
        lo, hi = verts.min(0), verts.max(0)
        size = hi - lo
        centre = 0.5 * (lo + hi)
        if kind == 'cylinder':
            # a 24-sided convex prism (flat base, contact on a face): Bullet's pure cylinder jitters and walks
            r = 0.5 * float(max(size[0], size[1]))
            a = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
            ring = np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]
            v = np.r_[np.c_[ring, np.full(24, lo[2])], np.c_[ring, np.full(24, hi[2])]]
            tris = []
            for k in range(24):
                n = (k + 1) % 24
                tris += [k, n, 24 + n, k, 24 + n, 24 + k]
            for k in range(1, 23):
                tris += [0, k + 1, k, 24, 24 + k, 24 + k + 1]
            pure = Shape.create_mesh(v.flatten().tolist(), tris)
            pure.set_respondable(True)
            pure.set_renderable(False)
            pure.set_dynamic(False)
            try:                                       # CoppeliaSim detects convexity by itself; make sure
                sim.simSetObjectInt32Parameter(pure.get_handle(), sim.sim_shapeintparam_convex, 1)
            except Exception:
                pass
        else:
            pure = Shape.create(type=PrimitiveShape.CUBOID, size=[float(x) for x in size], mass=0.1,
                                respondable=True, static=True, renderable=False, position=list(centre))
        visuals = [s for s in self.shape.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        for v in visuals:
            v.set_parent(pure, keep_in_place=True)
        self.shape.remove()
        self.shape = pure

    # --- collision shape of the cover cup ---

    def _make_hollow(self, n: int = 12, wall: float = 0.005) -> None:
        """Replaces the solid convex collision shape with an open shell of pure shapes (a ring of thin walls and the
        closed end), so that the upside-down cup can be put over a small object."""
        verts = _mesh_world(self.shape)
        lo, hi = verts.min(0), verts.max(0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        h = hi[2] - lo[2]
        r = np.hypot(verts[:, 0] - cx, verts[:, 1] - cy)
        r_low = float(r[verts[:, 2] < lo[2] + 0.1 * h].max())     # rim (on the table)
        r_high = float(r[verts[:, 2] > hi[2] - 0.1 * h].max())    # closed end (top)
        taper = np.arctan2(r_low - r_high, h)
        slant = h / np.cos(taper)
        parts = []
        for k in range(n):
            a = 2 * np.pi * k / n
            rm = 0.5 * (r_low + r_high) - 0.5 * wall
            width = 2 * (0.5 * (r_low + r_high)) * np.tan(np.pi / n) + 0.004
            p = Shape.create(type=PrimitiveShape.CUBOID, size=[wall, width, slant - 0.004], mass=0.01,
                             respondable=True, static=True, renderable=False,
                             position=[cx + rm * np.cos(a), cy + rm * np.sin(a), lo[2] + 0.5 * h],
                             orientation=[0.0, 0.0, 0.0])
            # tilt the wall inwards (taper) about its tangent axis, then turn it to its angle
            rot = np.eye(4)
            rot[:3, :3] = _euler_matrix((0.0, 0.0, a)) @ _euler_matrix((0.0, -taper, 0.0))
            rot[:3, 3] = p.get_position()
            p.set_matrix(rot)
            parts.append(p)
        base = 3 * wall                                       # the cup's base is thicker than its wall
        top = Shape.create(type=PrimitiveShape.CYLINDER, size=[2 * r_high, 2 * r_high, base], mass=0.01,
                           respondable=True, static=True, renderable=False,
                           position=[cx, cy, hi[2] - 0.5 * base])
        parts.append(top)
        shell = Shape(sim.simGroupShapes([p.get_handle() for p in parts]))
        visuals = [s for s in self.shape.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        for v in visuals:
            v.set_parent(shell, keep_in_place=True)
        self.shape.remove()
        shell.set_renderable(False)
        self.shape = shell

    # --- pose -------------------------------------------------------------------------------------------------------

    def set_pose(self, position: Sequence[float], quaternion: Sequence[float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Item frame at position / quaternion [x, y, z, w] (a dynamic item is teleported and stops)."""
        world = np.eye(4)
        world[:3, :3] = quat_to_matrix(quaternion)
        world[:3, 3] = position
        self.set_frame(world)

    def set_frame(self, world: np.ndarray) -> None:
        self.shape.set_matrix(world @ self._local)
        if self.state == 'passive':
            self.shape.reset_dynamic_object()

    def frame(self) -> np.ndarray:
        return self.shape.get_matrix() @ np.linalg.inv(self._local)

    def position(self) -> np.ndarray:
        return self.frame()[:3, 3].copy()

    def quaternion(self) -> np.ndarray:
        return matrix_to_quat(self.frame())

    def yaw(self) -> float:
        m = self.frame()
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def tilt(self) -> float:
        """Angle [rad] between the item's up axis and the world z axis (0: standing as it should)."""
        return float(np.arccos(np.clip(self.frame()[2, 2], -1.0, 1.0)))

    def top_z(self) -> float:
        return float(_mesh_world(self.shape)[:, 2].max())

    def velocity(self) -> np.ndarray:
        lin, _ = sim.simGetObjectVelocity(self.shape.get_handle())
        return np.asarray(lin, dtype=np.float64)

    # --- state ------------------------------------------------------------------------------------------------------

    def show(self) -> None:
        self.shape.set_respondable(True)
        self.shape.set_collidable(True)
        for v in self.visuals:
            v.set_renderable(True)
        if getattr(self, '_self_visual', False):
            self.shape.set_renderable(True)

    def hide(self) -> None:
        """Out of the scene: kinematic, not respondable, parked under the floor."""
        self.make_kinematic()
        self.shape.set_respondable(False)
        self.shape.set_collidable(False)
        for v in self.visuals:
            v.set_renderable(False)
        if getattr(self, '_self_visual', False):
            self.shape.set_renderable(False)
        park = PARK_POSITION + np.random.default_rng(zlib.crc32(self.name.encode())).uniform(-0.5, 0.5, 3)
        self.set_pose(park)
        self.state = 'hidden'

    @property
    def hidden(self) -> bool:
        return self.state == 'hidden'

    def make_passive(self) -> None:
        """A free rigid body (dynamics on, at rest)."""
        self.show()
        self.shape.set_dynamic(True)
        self.shape.reset_dynamic_object()
        self.state = 'passive'

    def make_kinematic(self) -> None:
        if self.shape.is_dynamic():
            self.shape.set_dynamic(False)
            self.shape.reset_dynamic_object()

    def make_carried(self) -> None:
        """Held by the person: kinematic, still respondable (it pushes free objects it runs into)."""
        self.show()
        self.make_kinematic()
        self.state = 'carried'

    def place(self, position: Sequence[float], yaw: float = 0.0, passive: bool = True) -> None:
        """Stands the item at position (bottom centre) with the given yaw, as a free body (or kinematic)."""
        self.show()
        self.state = 'carried'
        self.make_kinematic()
        self.set_pose(position, yaw_quat(yaw))
        if passive:
            self.make_passive()

    def handles(self) -> List[int]:
        return [self.shape.get_handle()] + [v.get_handle() for v in self.visuals]

    def remove(self) -> None:
        remove_tree(self.shape)


class TaskObject(Item):
    """An object of the static task (a jar, the jar lid, a block, a grocery item) seen through the Item interface, so
    that the person can carry it. Its frame: origin at the bottom centre of its bounding box and z up, yaw of the
    shape, as it stands when wrapped (the task's layout)."""

    def __init__(self, shape: Shape):
        self.kind = 'task_object'
        self.shape = shape
        self.name = shape.get_name()
        self.visuals = [s for s in shape.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        self.mass = float(shape.get_mass())
        self._was_dynamic = shape.is_dynamic()
        self.rewrap()

    def rewrap(self) -> None:
        """Recomputes the frame from the current (standing) pose, and remembers that pose (the static task's)."""
        m = self.shape.get_matrix()
        self._start = m.copy()
        verts = _mesh_world(self.shape)
        lo, hi = verts.min(0), verts.max(0)
        yaw = float(np.arctan2(m[1, 0], m[0, 0]))
        frame = np.eye(4)
        frame[:3, :3] = quat_to_matrix(yaw_quat(yaw))
        frame[:3, 3] = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]]
        self._local = np.linalg.inv(frame) @ m
        self.half_extents = 0.5 * (hi - lo)
        self.height = float(hi[2] - lo[2])
        self.radius = float(np.linalg.norm(self.half_extents[:2]))
        self.state = 'passive' if self.shape.is_dynamic() else 'static'

    def hide(self) -> None:
        raise RuntimeError('task objects stay in the scene')

    def make_passive(self) -> None:
        """Let go: a free body again, or, for an object the static task keeps fixed (a jar), fixed where it is."""
        if self._was_dynamic:
            super().make_passive()
        else:
            self.show()
            self.state = 'static'

    def restore(self, pose: bool = False) -> None:
        """Back to the dynamics the static task gave it (after an episode in which the person carried it), and with
        pose=True to where the static task put it (for the static demo)."""
        if pose:
            self.shape.set_dynamic(False)
            self.shape.set_matrix(self._start)
        self.shape.set_dynamic(self._was_dynamic)
        if self._was_dynamic:
            self.shape.reset_dynamic_object()
        self.state = 'passive' if self._was_dynamic else 'static'


def remove_tree(obj) -> None:
    """Removes an object and everything below it (PyRep's remove() leaves the children of a non-model object)."""
    try:
        objs = obj.get_objects_in_tree(exclude_base=False)
    except Exception:
        return
    for o in reversed(objs):
        try:
            o.remove()
        except Exception:
            pass


def remove_named(name: str) -> None:
    """Removes a leftover object of that name (and its subtree), e.g. after an interrupted task load."""
    from pyrep.objects.object import Object
    if Object.exists(name):
        try:
            remove_tree(Object.get_object(name))
        except Exception:
            pass


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])
