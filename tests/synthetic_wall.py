"""
Build a synthetic climbing-wall GLB with known answers for testing extract_holds.py.

The wall is an overhang whose angle changes with height (-10 deg below 2 m, -15 deg
above), with its bottom at height 0, a strip of floor, other routes' holds baked into
the wall mesh, and copies of the route holds left inside the wall (as happens when
holds are duplicated in Blender). Route holds are separate objects stored with node
transforms (one with a rotation), like a Blender export. The room is on the +Z side.

Run directly to write tests/synthetic_wall.glb:
    python tests/synthetic_wall.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

BREAK_HEIGHT = 2.0
ANGLE_LOW = -10.0
ANGLE_HIGH = -15.0
ROOM = np.array([0.0, 0.0, 1.0])
UP = np.array([0.0, 1.0, 0.0])
NODE_OFFSET = np.array([0.9956, 0.0, -0.259])  # Blender-like object location


@dataclass
class HoldSpec:
    """
    A route hold extruded from a side profile, with its expected answers.
    """

    name: str
    x: float
    y: float
    width: float
    depth: float
    tilt_deg: float  # grip surface tilt, positive = incut
    color: str | None  # "texture", "vertex" or None
    rotated_node: bool = False


HOLDS = [
    HoldSpec("foot_chip", x=0.30, y=0.25, width=0.06, depth=0.025, tilt_deg=0.0, color=None),
    HoldSpec("jug_start", x=-0.40, y=1.00, width=0.10, depth=0.06, tilt_deg=20.0, color="texture"),
    HoldSpec("ledge", x=0.50, y=1.50, width=0.25, depth=0.03, tilt_deg=0.0, color=None, rotated_node=True),
    HoldSpec("sloper_top", x=0.00, y=3.00, width=0.15, depth=0.07, tilt_deg=-25.0, color="vertex"),
]

# Other routes' holds, baked into the wall mesh: (x, y, radius).
OTHER_HOLDS = [(-0.25, 1.05, 0.05), (0.62, 1.40, 0.06), (0.15, 3.10, 0.07), (0.45, 0.30, 0.04), (-0.9, 2.5, 0.08)]


def wall_angle_at(y: float) -> float:
    """
    Signed wall angle (degrees, negative = overhang) at a height.

    Args:
        y (float): Height in meters.

    Returns:
        float: Wall angle in degrees.
    """
    return ANGLE_LOW if y < BREAK_HEIGHT else ANGLE_HIGH


def wall_z(y: np.ndarray | float) -> np.ndarray | float:
    """
    Depth of the wall surface (toward the room, +Z) at a height.

    Args:
        y (np.ndarray | float): Height(s) in meters.

    Returns:
        np.ndarray | float: Z coordinate(s) of the wall surface.
    """
    lo, hi = math.tan(math.radians(-ANGLE_LOW)), math.tan(math.radians(-ANGLE_HIGH))
    y = np.asarray(y, dtype=float)
    return np.where(y < BREAK_HEIGHT, lo * y, lo * BREAK_HEIGHT + hi * (y - BREAK_HEIGHT))


def wall_frame_at(y: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Wall normal (into the room) and up-the-wall direction at a height.

    Args:
        y (float): Height in meters.

    Returns:
        tuple[np.ndarray, np.ndarray]: (normal, along) unit vectors.
    """
    a = math.radians(abs(wall_angle_at(y)))
    normal = np.array([0.0, -math.sin(a), math.cos(a)])
    along = np.array([0.0, math.cos(a), math.sin(a)])
    return normal, along


def unweld(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """
    Give every triangle its own three vertices, like a Polycam export.

    Args:
        vertices (np.ndarray): Vertex positions (m, 3).
        faces (np.ndarray): Triangles (n, 3).

    Returns:
        trimesh.Trimesh: Unwelded mesh.
    """
    tri = vertices[faces].reshape(-1, 3)
    return trimesh.Trimesh(tri, np.arange(len(tri)).reshape(-1, 3), process=False)


def grid(xs: np.ndarray, ys: np.ndarray, to_3d) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a rectangular parameter grid.

    Args:
        xs (np.ndarray): First parameter samples.
        ys (np.ndarray): Second parameter samples.
        to_3d (callable): Maps parameter arrays (X, Y) to points (n, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Vertices and faces.
    """
    X, Y = np.meshgrid(xs, ys)
    verts = to_3d(X.ravel(), Y.ravel())
    nx = len(xs)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(len(ys) - 1))
    a = (j * nx + i).ravel()
    faces = np.vstack([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)])
    return verts, faces


def bump(x: float, y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    A flattened half-sphere sitting on the wall, used for other routes' holds.

    Args:
        x (float): Horizontal position.
        y (float): Height.
        radius (float): Radius in meters.

    Returns:
        tuple[np.ndarray, np.ndarray]: Vertices and faces.
    """
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    v = sphere.vertices.copy()
    normal, along = wall_frame_at(y)
    side = np.cross(along, normal)
    keep = v[:, 2] >= -1e-9
    local = v[:, 0, None] * side + v[:, 1, None] * along + 0.6 * np.maximum(v[:, 2], 0)[:, None] * normal
    base = np.array([x, y, float(wall_z(y))])
    faces = sphere.faces[keep[sphere.faces].all(1)]
    return base + local, faces


def profile_hold(spec: HoldSpec) -> tuple[trimesh.Trimesh, dict]:
    """
    Build a route hold by extruding a side profile along the wall's horizontal.

    The profile (in the vertical plane through the wall normal) is: bottom on the
    wall, top-back on the wall, top-front (grip surface tilted by tilt_deg), and a
    front face dropping straight down. The back (wall contact) is left open.

    Args:
        spec (HoldSpec): Hold description.

    Returns:
        tuple[trimesh.Trimesh, dict]: World-space mesh (welded, subdivided) and expected answers.
    """
    normal, along = wall_frame_at(spec.y)
    anchor = np.array([spec.x, spec.y, float(wall_z(spec.y))])
    theta = math.radians(spec.tilt_deg)
    top_back = anchor + 0.03 * along
    bottom = anchor - 0.03 * along
    top_front = top_back + spec.depth * (math.cos(theta) * ROOM + math.sin(theta) * UP)
    front_bottom = top_front - (top_front - bottom) @ UP * UP * 0.7
    profile = [bottom, top_back, top_front, front_bottom]
    half = np.array([spec.width / 2, 0, 0])
    left = [p - half for p in profile]
    right = [p + half for p in profile]
    verts = np.array(left + right)
    quads = [(1, 2), (2, 3), (3, 0)]  # top, front, underside; the back (0, 1) stays open
    faces = []
    for a, b in quads:
        faces += [[a, b, b + 4], [a, b + 4, a + 4]]
    faces += [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]  # end caps
    v, f = trimesh.remesh.subdivide_to_size(verts, np.array(faces), max_edge=0.008)
    mesh = trimesh.Trimesh(v, f, process=True)

    expected = {
        "wall_angle_deg": wall_angle_at(spec.y),
        "position": (mesh.triangles_center * mesh.area_faces[:, None]).sum(0) / mesh.area,
        "protrusion_m": float(((verts - anchor) @ normal).max()),
        "incut_angle_deg": spec.tilt_deg,
        "slope_deg": abs(spec.tilt_deg),
        "edge_depth_m": float((top_front - top_back) @ normal),
        "footprint_w": spec.width,
        "two_hands": spec.width >= 0.12,
    }
    return mesh, expected


def add_color(mesh: trimesh.Trimesh, kind: str | None) -> trimesh.Trimesh:
    """
    Attach a known color as a texture or as vertex colors.

    Args:
        mesh (trimesh.Trimesh): Unwelded mesh.
        kind (str | None): "texture", "vertex" or None.

    Returns:
        trimesh.Trimesh: The same mesh with its visual set.
    """
    if kind == "texture":
        image = Image.new("RGB", (8, 8), (200, 30, 30))
        uv = np.full((len(mesh.vertices), 2), 0.5)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    elif kind == "vertex":
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile([30, 180, 60, 255], (len(mesh.vertices), 1)))
    return mesh


def build(
    path: str | Path,
    wall_name: str = "Wall_scan",
    flip_fraction: float = 0.3,
    up: str = "y",
    invert_wall: bool = False,
    seed: int = 0,
) -> dict:
    """
    Write the synthetic GLB and return the expected answers.

    Args:
        path (str | Path): Output GLB path.
        wall_name (str): Node name for the wall object.
        flip_fraction (float): Fraction of the sloper's faces whose winding is flipped.
        up (str): "y" (glTF convention) or "z" (whole scene rotated so +Z is up).
        invert_wall (bool): Flip the winding of every wall face (normals point into the wall).
        seed (int): Random seed for noise and flips.

    Returns:
        dict: {hold name: expected answers}, positions in the file's frame.
    """
    rng = np.random.default_rng(seed)
    to_file = np.eye(4) if up == "y" else trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])

    # Wall surface with 1.5 mm scan noise, plus floor in front of it.
    wall_v, wall_f = grid(
        np.arange(-1.5, 1.5001, 0.02),
        np.arange(0.0, 4.0001, 0.02),
        lambda x, y: np.stack([x, y, wall_z(y) + rng.normal(0, 0.0015, len(x))], 1),
    )
    floor_v, floor_f = grid(
        np.arange(-1.5, 1.5001, 0.04),
        np.arange(0.0, 1.5001, 0.04),
        lambda x, z: np.stack([x, rng.normal(0, 0.0015, len(x)), z], 1),
    )
    parts = [unweld(wall_v, wall_f), unweld(floor_v, floor_f)]
    parts += [unweld(*bump(x, y, r)) for x, y, r in OTHER_HOLDS]

    expected, holds = {}, []
    for spec in HOLDS:
        mesh, exp = profile_hold(spec)
        expected[spec.name] = exp
        parts.append(unweld(mesh.vertices, mesh.faces))  # copy left in the wall
        hold = unweld(mesh.vertices, mesh.faces)
        if flip_fraction and spec.color == "vertex":
            flip = rng.random(len(hold.faces)) < flip_fraction
            hold.faces[flip] = hold.faces[flip][:, ::-1]
        holds.append((spec, add_color(hold, spec.color)))

    wall = trimesh.util.concatenate(parts)
    if invert_wall:
        wall.faces = wall.faces[:, ::-1]
    wall.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(wall.vertices), 2), 0.5), image=Image.new("RGB", (8, 8), (128, 128, 128))
    )

    scene = trimesh.Scene()
    offset = trimesh.transformations.translation_matrix(NODE_OFFSET)

    def add(mesh: trimesh.Trimesh, name: str, node_transform: np.ndarray) -> None:
        """
        Add a mesh whose vertices are stored in node-local coordinates.

        Args:
            mesh (trimesh.Trimesh): Mesh in world (Y-up) coordinates.
            name (str): Node and geometry name.
            node_transform (np.ndarray): Node transform in the Y-up frame (4, 4).

        Returns:
            None
        """
        world = to_file @ node_transform
        local = mesh.copy()
        local.apply_transform(np.linalg.inv(node_transform))
        scene.add_geometry(local, node_name=name, geom_name=name + "_mesh", transform=world)

    add(wall, wall_name, offset)
    for spec, hold in holds:
        T = offset
        if spec.rotated_node:
            T = offset @ trimesh.transformations.rotation_matrix(math.radians(35), [0.3, 1, 0.2])
        add(hold, spec.name, T)
    scene.export(str(path), file_type="glb")

    if up == "z":
        R = to_file[:3, :3]
        for exp in expected.values():
            exp["position"] = R @ exp["position"]
    return expected


if __name__ == "__main__":
    out = Path(__file__).with_name("synthetic_wall.glb")
    build(out)
    print(f"wrote {out}")
