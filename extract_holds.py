"""
Extract per-hold position and geometric features from a climbing-wall GLB.

The GLB must contain one wall object (a LiDAR scan, which may still include other
routes' holds and part of the floor) plus the holds of ONE route as separate mesh
objects. For each hold, a wall plane is fitted locally (RANSAC on nearby wall
points) and every feature is measured relative to that plane.

Usage:
    python extract_holds.py route.glb --out route_holds.json [--plot]

Outputs:
    <out>.json   holds and metadata (schema_version 0.1, see README of DeMoSearch)
    <out>.csv    one row per hold, sorted by height_m (path set with --csv)
    stdout       a summary table and any warnings

Coordinate conventions:
    All coordinates are in the GLB's world frame (node transforms applied), in
    meters. glTF is Y-up, so --up defaults to y. wall_normal points into the room.
    wall_angle_deg: 0 = vertical, negative = overhang, positive = slab.
    incut_angle_deg: positive = grip surface tilted back toward the wall (incut),
    negative = tilted away from the wall (sloping).
    mean_color is RGB in 0-255.

Parameters (command-line flags, defaults in brackets):
    --out PATH              JSON output path [<input stem>_holds.json]
    --csv PATH              CSV output path [JSON path with .csv suffix]
    --wall NAME             wall object (node or mesh name). Default: first object
                            whose name contains "wall", else the largest-area mesh.
    --up {y,z}              up axis of the file [y]
    --radius M              base radius of wall points used for the local plane;
                            half the hold's largest extent is added to it [0.35]
    --exclude-margin M      wall faces closer than this to any route hold are
                            ignored for plane fitting (removes the holds' own
                            copies and their bases) [0.03]
    --ransac-iters N        RANSAC iterations per hold [500]
    --ransac-thresh M       RANSAC inlier distance [0.01]
    --max-wall-angle DEG    reject candidate planes tilted more than this from
                            vertical, e.g. the floor [80]
    --max-gap M             a candidate plane must pass within this distance of
                            the hold (the hold touches its wall) [0.04]
    --straddle-frac F       reject candidate planes with more than this fraction
                            of hold vertices farther than --max-gap on BOTH
                            sides, i.e. planes cutting through the hold [0.1]
    --grip-max-angle DEG    faces whose normal is within this angle of "up" form
                            the grip surface [60]
    --vertical-tol DEG      |wall_angle_deg| <= this is labelled "vertical" [2]
    --two-hands-width M     footprint width needed for two hands [0.12]
    --bottom-percentile P   wall bottom = this percentile of the wall's heights,
                            robust to scan noise; the raw minimum is also
                            reported [0.5]
    --bottom-warn M         warn if the wall bottom is farther than this from 0 [0.05]
    --weld-tol M            vertices closer than this are merged before checking
                            connectivity/winding [1e-5]
    --flat-tol M            hold faces within this distance of the plane and
                            parallel to it count as "wall-like" [0.008]
    --flat-frac F           flag suspiciously_flat if wall-like area fraction >= F [0.5]
    --min-protrusion M      flag suspiciously_flat if protrusion is below this [0.015]
    --min-faces N           flag few_faces below this face count [30]
    --min-grip-area M2      flag no_grip_surface below this grip area [2e-4]
    --behind-tol M          flag points_behind_wall if the hold extends more than
                            this behind the plane [0.02]
    --seed N                RNG seed for RANSAC [0]
    --plot                  show a 3D matplotlib view of wall, holds and arrows
    --plot-out PATH         save that view as an image instead of showing it

Quality flags per hold:
    no_grip_surface, few_faces, suspiciously_flat, points_behind_wall,
    multiple_parts, plane_fit_failed, sparse_wall, normal_sign_ambiguous,
    normal_sign_conflict, no_color

Dependencies: numpy, scipy, trimesh, pillow (texture colors), matplotlib (--plot only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import cKDTree

SCHEMA_VERSION = "0.1"
UP_VECTORS = {"y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


@dataclass
class Params:
    """
    Tunable thresholds for hold extraction. See the module docstring for meanings.
    """

    wall: str | None = None
    up: str = "y"
    radius: float = 0.35
    exclude_margin: float = 0.03
    ransac_iters: int = 500
    ransac_thresh: float = 0.01
    max_wall_angle: float = 80.0
    max_gap: float = 0.04
    straddle_frac: float = 0.1
    grip_max_angle: float = 60.0
    vertical_tol: float = 2.0
    two_hands_width: float = 0.12
    bottom_percentile: float = 0.5
    bottom_warn: float = 0.05
    weld_tol: float = 1e-5
    flat_tol: float = 0.008
    flat_frac: float = 0.5
    min_protrusion: float = 0.015
    min_faces: int = 30
    min_grip_area: float = 2e-4
    behind_tol: float = 0.02
    seed: int = 0


@dataclass
class SceneObject:
    """
    One mesh object from the GLB, with its node transform already applied.
    """

    name: str
    geom_name: str
    mesh: trimesh.Trimesh


@dataclass
class Plane:
    """
    A local wall plane: a point on it, a unit normal into the room, and fit info.
    """

    origin: np.ndarray
    normal: np.ndarray
    n_inliers: int
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_objects(path: str | Path) -> list[SceneObject]:
    """
    Load every mesh node of a GLB in world coordinates.

    Args:
        path (str | Path): Path to the GLB file.

    Returns:
        list[SceneObject]: One entry per mesh node, transform applied.
    """
    scene = trimesh.load(str(path), force="scene")
    objects = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        # Build a new mesh instead of copy() so the (large) texture image is shared.
        mesh = trimesh.Trimesh(
            vertices=trimesh.transform_points(geom.vertices, transform),
            faces=geom.faces,
            visual=geom.visual,
            process=False,
        )
        objects.append(SceneObject(name=str(node), geom_name=str(geom_name), mesh=mesh))
    return objects


def pick_wall(objects: list[SceneObject], wall_name: str | None) -> tuple[SceneObject, str]:
    """
    Choose the wall object by explicit name, by "wall" in the name, or by largest area.

    Args:
        objects (list[SceneObject]): All mesh objects in the file.
        wall_name (str | None): Name given with --wall, or None.

    Returns:
        tuple[SceneObject, str]: The wall object and a sentence explaining the choice.
    """
    if wall_name is not None:
        for obj in objects:
            if wall_name in (obj.name, obj.geom_name):
                return obj, f"wall '{obj.name}' chosen by --wall"
        raise SystemExit(f"error: no object named '{wall_name}'. Objects: {[o.name for o in objects]}")
    named = [o for o in objects if "wall" in o.name.lower() or "wall" in o.geom_name.lower()]
    if named:
        wall = max(named, key=lambda o: o.mesh.area)
        return wall, f"wall '{wall.name}' chosen because its name contains 'wall'"
    wall = max(objects, key=lambda o: o.mesh.area)
    return wall, f"no object name contains 'wall'; using the largest-area object '{wall.name}' as the wall"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (or rows of a matrix) to unit length, leaving zeros as zeros.

    Args:
        v (np.ndarray): Vector of shape (3,) or array of shape (n, 3).

    Returns:
        np.ndarray: Normalized array of the same shape.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v, dtype=float), where=norm > 1e-12)


def weld(mesh: trimesh.Trimesh, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge vertices that share a position (scan meshes often store 3 vertices per face).

    Args:
        mesh (trimesh.Trimesh): Input mesh.
        tol (float): Positions are compared after rounding to this step (meters).

    Returns:
        tuple[np.ndarray, np.ndarray]: Welded vertices (m, 3) and faces (n, 3), same face order.
    """
    key = np.round(mesh.vertices / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return mesh.vertices[first], inverse.reshape(-1)[mesh.faces]


def triangle_normals_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unit normals (from winding) and areas of triangles.

    Args:
        vertices (np.ndarray): Vertex positions (m, 3).
        faces (np.ndarray): Triangle vertex indices (n, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Unit normals (n, 3) and areas (n,).
    """
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return unit(cross), 0.5 * np.linalg.norm(cross, axis=1)


def consistent_winding(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Make triangle winding consistent inside each connected component.

    Scans can contain faces with flipped normals; neighbors that traverse their
    shared edge in the same direction disagree, and one of them is flipped.

    Args:
        faces (np.ndarray): Welded triangle indices (n, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Re-wound faces (n, 3) and a component label per face.
    """
    n = len(faces)
    adjacency, edges = trimesh.graph.face_adjacency(faces, return_edges=True)
    if len(adjacency) == 0:
        return faces.copy(), np.arange(n)

    def has_directed_edge(face_idx: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Check whether each face contains the directed edge a->b.

        Args:
            face_idx (np.ndarray): Face indices (k,).
            a (np.ndarray): Edge start vertices (k,).
            b (np.ndarray): Edge end vertices (k,).

        Returns:
            np.ndarray: Boolean array (k,).
        """
        f = faces[face_idx]
        return ((f[:, 0] == a) & (f[:, 1] == b)) | ((f[:, 1] == a) & (f[:, 2] == b)) | ((f[:, 2] == a) & (f[:, 0] == b))

    d0 = has_directed_edge(adjacency[:, 0], edges[:, 0], edges[:, 1])
    d1 = has_directed_edge(adjacency[:, 1], edges[:, 0], edges[:, 1])
    disagree = (d0 == d1).astype(np.int8)
    # Store disagreement + 1 so that consistent pairs are still graph edges.
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    vals = np.concatenate([disagree, disagree]) + 1
    graph = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)

    flip = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        order, pred = breadth_first_order(graph, start, directed=False, return_predecessors=True)
        visited[order] = True
        for node in order[1:]:
            parent = pred[node]
            flip[node] = flip[parent] ^ bool(graph[parent, node] - 1)
    out = faces.copy()
    out[flip] = out[flip][:, ::-1]
    return out, labels


# ---------------------------------------------------------------------------
# Local wall plane
# ---------------------------------------------------------------------------


def weighted_plane(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares plane through weighted points.

    Args:
        points (np.ndarray): Points (n, 3).
        weights (np.ndarray): Non-negative weights (n,).

    Returns:
        tuple[np.ndarray, np.ndarray]: Plane origin (3,) and unit normal (3,), sign arbitrary.
    """
    w = weights / weights.sum()
    origin = (points * w[:, None]).sum(0)
    centered = (points - origin) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return origin, vt[2]


def fit_local_plane(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    hold_vertices: np.ndarray,
    center: np.ndarray,
    radius: float,
    up: np.ndarray,
    params: Params,
    rng: np.random.Generator,
) -> Plane:
    """
    Fit the wall plane around one hold with RANSAC, and orient its normal into the room.

    Candidate planes must be within --max-wall-angle of vertical (rejects the floor)
    and pass within --max-gap of the hold (rejects other walls and volumes behind).
    Inliers are scored by area times a Gaussian weight on distance to the hold.

    The normal's sign comes from the scan's own face normals (which face the
    scanner, i.e. the room); when those are inconsistent, from the side the hold
    protrudes to. Disagreement between the two is flagged.

    Args:
        points (np.ndarray): Wall face centers near the hold (n, 3).
        normals (np.ndarray): Scan face normals of those faces (n, 3).
        areas (np.ndarray): Areas of those faces (n,).
        hold_vertices (np.ndarray): The hold's vertices (m, 3).
        center (np.ndarray): The hold's position (3,).
        radius (float): Neighborhood radius used, for the locality weight.
        up (np.ndarray): Unit up vector (3,).
        params (Params): Thresholds.
        rng (np.random.Generator): Random generator for RANSAC sampling.

    Returns:
        Plane: The oriented local wall plane.
    """
    locality = np.exp(-(np.linalg.norm(points - center, axis=1) / (0.5 * radius)) ** 2)
    weights = areas * locality
    max_tilt_sin = math.sin(math.radians(params.max_wall_angle))

    best_score, best_inliers = -1.0, None
    if len(points) >= 3:
        for _ in range(params.ransac_iters):
            sample = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            if abs(normal @ up) > max_tilt_sin:
                continue
            hold_dist = (hold_vertices - sample[0]) @ normal
            if np.abs(hold_dist).min() > params.max_gap:
                continue
            inliers = np.abs((points - sample[0]) @ normal) < params.ransac_thresh
            # The hold sits in front of its wall. When the scan normals say which side
            # is the room, the hold must not be behind the plane (this rejects the
            # neighboring panel at a fold); otherwise it must at least not straddle it.
            a = areas[inliers]
            scan_side = (normals[inliers] @ normal * a).sum() / (a.sum() + 1e-12)
            behind = (hold_dist < -params.max_gap).mean()
            in_front = (hold_dist > params.max_gap).mean()
            if abs(scan_side) >= 0.3:
                if (behind if scan_side > 0 else in_front) > params.straddle_frac:
                    continue
            elif min(behind, in_front) > params.straddle_frac:
                continue
            score = weights[inliers].sum()
            if score > best_score:
                best_score, best_inliers = score, inliers

    flags = []
    if best_inliers is None or best_inliers.sum() < 3:
        flags.append("plane_fit_failed")
        best_inliers = np.ones(len(points), dtype=bool)
    origin, normal = weighted_plane(points[best_inliers], weights[best_inliers])
    # One refinement pass with the least-squares plane.
    refined = np.abs((points - origin) @ normal) < params.ransac_thresh
    if refined.sum() >= 3:
        best_inliers = refined
        origin, normal = weighted_plane(points[best_inliers], weights[best_inliers])

    # Orientation votes: +1 means the current normal already points into the room.
    dist = (hold_vertices - origin) @ normal
    front, back = max(dist.max(), 0.0), max(-dist.min(), 0.0)
    hold_vote = (front - back) / (front + back + 1e-12)
    a = areas[best_inliers]
    scan_vote = float((normals[best_inliers] @ normal * a).sum() / (a.sum() + 1e-12))
    if abs(scan_vote) >= 0.3:
        sign = np.sign(scan_vote)
        if abs(hold_vote) >= 0.5 and np.sign(hold_vote) != sign:
            flags.append("normal_sign_conflict")
    elif abs(hold_vote) >= 0.2:
        sign = np.sign(hold_vote)
    else:
        sign = np.sign(hold_vote) if hold_vote != 0 else 1.0
        flags.append("normal_sign_ambiguous")
    return Plane(origin=origin, normal=normal * sign, n_inliers=int(best_inliers.sum()), flags=flags)


def wall_frame(normal: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontal and up-the-wall unit directions lying in the wall plane.

    Args:
        normal (np.ndarray): Wall normal into the room (3,).
        up (np.ndarray): Unit up vector (3,).

    Returns:
        tuple[np.ndarray, np.ndarray]: (horizontal direction, up-the-wall direction).
    """
    horizontal = np.cross(up, normal)
    if np.linalg.norm(horizontal) < 1e-6:  # roof or floor: any horizontal works
        horizontal = np.cross(up, np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    horizontal = unit(horizontal)
    along = unit(np.cross(normal, horizontal))
    return horizontal, along


def wall_angle(normal: np.ndarray, up: np.ndarray) -> float:
    """
    Signed wall angle from vertical: negative = overhang, positive = slab.

    Args:
        normal (np.ndarray): Wall normal into the room (3,).
        up (np.ndarray): Unit up vector (3,).

    Returns:
        float: Angle in degrees.
    """
    return math.degrees(math.asin(float(np.clip(normal @ up, -1.0, 1.0))))


def wall_kind(angle_deg: float, tolerance: float) -> str:
    """
    Label a wall angle as vertical, overhang or slab.

    Args:
        angle_deg (float): Signed wall angle in degrees.
        tolerance (float): Half-width of the "vertical" band in degrees.

    Returns:
        str: "vertical", "overhang" or "slab".
    """
    if abs(angle_deg) <= tolerance:
        return "vertical"
    return "overhang" if angle_deg < 0 else "slab"


# ---------------------------------------------------------------------------
# Colors and names
# ---------------------------------------------------------------------------


def mean_color(mesh: trimesh.Trimesh, areas: np.ndarray) -> list[int] | None:
    """
    Area-weighted mean RGB of a mesh from its texture or vertex/face colors.

    Args:
        mesh (trimesh.Trimesh): Mesh with its original visual (texture or colors).
        areas (np.ndarray): Face areas (n,).

    Returns:
        list[int] | None: [r, g, b] in 0-255, or None if the mesh has no color data.
    """
    visual = mesh.visual
    try:
        if visual.kind == "texture":
            material = visual.material
            image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            if image is None or visual.uv is None or len(visual.uv) != len(mesh.vertices):
                return None
            vertex_rgb = trimesh.visual.color.uv_to_color(visual.uv, image)[:, :3].astype(float)
            face_rgb = vertex_rgb[mesh.faces].mean(1)
        elif visual.kind == "vertex":
            face_rgb = np.asarray(visual.vertex_colors)[:, :3].astype(float)[mesh.faces].mean(1)
        elif visual.kind == "face":
            face_rgb = np.asarray(visual.face_colors)[:, :3].astype(float)
        else:
            return None
    except Exception as exc:  # malformed textures should not stop extraction
        print(f"warning: could not read colors: {exc}", file=sys.stderr)
        return None
    if areas.sum() <= 0:
        return None
    rgb = (face_rgb * areas[:, None]).sum(0) / areas.sum()
    return [int(round(c)) for c in rgb]


def name_tags(name: str) -> tuple[bool, bool]:
    """
    Detect start/top markers in a hold name.

    Args:
        name (str): Object name.

    Returns:
        tuple[bool, bool]: (is_start, is_top).
    """
    return bool(re.search("start", name, re.I)), bool(re.search("top", name, re.I))


# ---------------------------------------------------------------------------
# Per-hold features
# ---------------------------------------------------------------------------


def hold_features(
    obj: SceneObject,
    wall_points: np.ndarray,
    wall_normals: np.ndarray,
    wall_areas: np.ndarray,
    wall_tree: cKDTree,
    wall_usable: np.ndarray,
    wall_bottom: float,
    up: np.ndarray,
    params: Params,
    rng: np.random.Generator,
) -> dict:
    """
    Compute all features of one hold relative to its local wall plane.

    Args:
        obj (SceneObject): The hold.
        wall_points (np.ndarray): Wall face centers (N, 3).
        wall_normals (np.ndarray): Wall face normals from the scan (N, 3).
        wall_areas (np.ndarray): Wall face areas (N,).
        wall_tree (cKDTree): KD-tree over wall_points.
        wall_usable (np.ndarray): Boolean mask of wall faces allowed in plane fits (N,).
        wall_bottom (float): Height of the wall bottom along the up axis.
        up (np.ndarray): Unit up vector (3,).
        params (Params): Thresholds.
        rng (np.random.Generator): Random generator for RANSAC.

    Returns:
        dict: The hold's record in the output schema.
    """
    flags: list[str] = []
    vertices, faces = weld(obj.mesh, params.weld_tol)
    faces, components = consistent_winding(faces)
    face_normals, areas = triangle_normals_areas(vertices, faces)
    centers = vertices[faces].mean(1)
    position = (centers * areas[:, None]).sum(0) / areas.sum()
    if len(np.unique(components)) > 1:
        flags.append("multiple_parts")
    if len(faces) < params.min_faces:
        flags.append("few_faces")

    # Local wall points: grow the radius until enough usable points are found.
    radius = params.radius + 0.5 * float(np.ptp(vertices, axis=0).max())
    for _ in range(4):
        idx = np.asarray(wall_tree.query_ball_point(position, radius), dtype=int)
        idx = idx[wall_usable[idx]] if len(idx) else idx
        if len(idx) >= 50:
            break
        radius *= 1.5
        if "sparse_wall" not in flags:
            flags.append("sparse_wall")
    plane = fit_local_plane(
        wall_points[idx], wall_normals[idx], wall_areas[idx], vertices, position, radius, up, params, rng
    )
    flags += plane.flags
    n = plane.normal

    # Orient each connected part's faces so that they face the room overall.
    for label in np.unique(components):
        part = components == label
        if (face_normals[part] @ n * areas[part]).sum() < 0:
            face_normals[part] *= -1

    vertex_dist = (vertices - plane.origin) @ n
    face_dist = (centers - plane.origin) @ n
    protrusion = float(vertex_dist.max())
    if -vertex_dist.min() > params.behind_tol:
        flags.append("points_behind_wall")

    horizontal, along = wall_frame(n, up)
    rel = vertices - position
    footprint = [float(np.ptp(rel @ horizontal)), float(np.ptp(rel @ along))]

    wall_like = (np.abs(face_dist) < params.flat_tol) & (face_normals @ n > 0.95)
    flat_frac = float(areas[wall_like].sum() / areas.sum())
    if flat_frac >= params.flat_frac or protrusion < params.min_protrusion:
        flags.append("suspiciously_flat")

    grip = (face_normals @ up >= math.cos(math.radians(params.grip_max_angle))) & ~wall_like
    grip_area = float(areas[grip].sum())
    facing = slope = incut = edge_depth = None
    if grip_area >= params.min_grip_area:
        facing_vec = unit((face_normals[grip] * areas[grip, None]).sum(0))
        facing = facing_vec
        slope = math.degrees(math.acos(float(np.clip(facing_vec @ up, -1, 1))))
        out_dir = n - (n @ up) * up
        out_dir = unit(out_dir) if np.linalg.norm(out_dir) > 1e-6 else -along
        incut = math.degrees(math.atan2(-float(facing_vec @ out_dir), float(facing_vec @ up)))
        grip_dist = vertex_dist[np.unique(faces[grip])]
        edge_depth = float(grip_dist.max() - grip_dist.min())
    else:
        flags.append("no_grip_surface")

    angle = wall_angle(n, up)
    color = mean_color(obj.mesh, triangle_normals_areas(obj.mesh.vertices, obj.mesh.faces)[1])
    if color is None:
        flags.append("no_color")
    is_start, is_top = name_tags(obj.name)

    def vec(v: np.ndarray | None) -> list[float] | None:
        """
        Round a vector for JSON output.

        Args:
            v (np.ndarray | None): Vector or None.

        Returns:
            list[float] | None: Rounded list, or None.
        """
        return None if v is None else [round(float(x), 5) for x in v]

    def num(x: float | None, digits: int) -> float | None:
        """
        Round a number for JSON output.

        Args:
            x (float | None): Value or None.
            digits (int): Decimal places.

        Returns:
            float | None: Rounded value, or None.
        """
        return None if x is None else round(float(x), digits)

    return {
        "id": obj.name,
        "position": vec(position),
        "height_m": num(position @ up - wall_bottom, 4),
        "facing": vec(facing),
        "wall_normal": vec(n),
        "wall_angle_deg": num(angle, 2),
        "wall_kind": wall_kind(angle, params.vertical_tol),
        "protrusion_m": num(protrusion, 4),
        "footprint_m": [num(footprint[0], 4), num(footprint[1], 4)],
        "grip_area_m2": num(grip_area, 6),
        "slope_deg": num(slope, 2),
        "incut_angle_deg": num(incut, 2),
        "edge_depth_m": num(edge_depth, 4),
        "two_hands": bool(footprint[0] >= params.two_hands_width),
        "mean_color": color,
        "is_start": is_start,
        "is_top": is_top,
        "flags": flags,
        "n_faces": int(len(faces)),
        "surface_area_m2": num(areas.sum(), 6),
        "wall_like_area_frac": num(flat_frac, 3),
        "plane_inliers": plane.n_inliers,
    }


# ---------------------------------------------------------------------------
# Whole-file extraction
# ---------------------------------------------------------------------------


def extract(path: str | Path, params: Params) -> tuple[dict, dict]:
    """
    Run hold extraction on a GLB file.

    Args:
        path (str | Path): Path to the GLB file.
        params (Params): Thresholds and options.

    Returns:
        tuple[dict, dict]: The output document (JSON schema) and a context dict with
        the loaded wall and hold meshes (for plotting).
    """
    up = UP_VECTORS[params.up]
    objects = load_objects(path)
    if not objects:
        raise SystemExit("error: the file contains no mesh objects")
    if len(objects) == 1:
        raise SystemExit(
            "error: the file contains only one mesh, so the holds are not separate objects.\n"
            "In Blender: select the wall, Tab into Edit Mode, select each hold's faces, press P > Selection,\n"
            "then Tab out and re-export the GLB."
        )
    wall, reason = pick_wall(objects, params.wall)
    print(reason)
    holds = [o for o in objects if o is not wall]
    empty = [o.name for o in holds if len(o.mesh.faces) == 0]
    for name in empty:
        print(f"warning: hold object '{name}' has no faces; skipped", file=sys.stderr)
    holds = [o for o in holds if len(o.mesh.faces) > 0]

    heights = wall.mesh.vertices @ up
    raw_min = float(heights.min())
    bottom = float(np.percentile(heights, params.bottom_percentile))
    print(f"wall bottom along {params.up}: {bottom:.4f} m ({params.bottom_percentile} percentile), raw minimum {raw_min:.4f} m")
    if abs(bottom) > params.bottom_warn:
        print(f"warning: wall bottom is {bottom:.3f} m from 0 along {params.up}; is --up correct?", file=sys.stderr)

    wall_normals, wall_areas = triangle_normals_areas(wall.mesh.vertices, wall.mesh.faces)
    wall_points = wall.mesh.triangles_center
    wall_tree = cKDTree(wall_points)
    hold_points = np.vstack([np.vstack([h.mesh.vertices, h.mesh.triangles_center]) for h in holds])
    near_hold, _ = cKDTree(hold_points).query(wall_points, distance_upper_bound=params.exclude_margin)
    wall_usable = ~np.isfinite(near_hold)

    def run_all(normals: np.ndarray) -> list[dict]:
        """
        Compute every hold's record with the given wall face normals.

        Args:
            normals (np.ndarray): Wall face normals (N, 3).

        Returns:
            list[dict]: Hold records.
        """
        rng = np.random.default_rng(params.seed)
        return [
            hold_features(h, wall_points, normals, wall_areas, wall_tree, wall_usable, bottom, up, params, rng)
            for h in holds
        ]

    records = run_all(wall_normals)
    # If most holds protrude against the scan normals, the scan's winding is inverted.
    conflicts = sum("normal_sign_conflict" in r["flags"] for r in records)
    if conflicts > len(records) / 2:
        print("warning: the wall's face normals point away from the holds; treating the scan winding as inverted",
              file=sys.stderr)
        records = run_all(-wall_normals)
    records.sort(key=lambda r: r["height_m"])
    starts = [r["id"] for r in records if r["is_start"]]
    tops = [r["id"] for r in records if r["is_top"]]
    if len(tops) > 1:
        print(f"warning: several top holds {tops}; using the highest", file=sys.stderr)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "units": "meters",
        "up_axis": params.up,
        "source_file": Path(path).name,
        "wall_object": wall.name,
        "wall_bottom_height": round(bottom, 4),
        "wall_bottom_raw_min": round(raw_min, 4),
        "params": {f.name: getattr(params, f.name) for f in fields(params)},
        "holds": records,
        "start_hold_ids": starts,
        "top_hold_id": tops[-1] if tops else None,
    }
    return doc, {"wall": wall, "holds": holds}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "id", "height_m", "x", "y", "z", "wall_angle_deg", "wall_kind", "protrusion_m",
    "footprint_w_m", "footprint_h_m", "grip_area_m2", "slope_deg", "incut_angle_deg",
    "edge_depth_m", "two_hands", "facing_x", "facing_y", "facing_z", "normal_x",
    "normal_y", "normal_z", "r", "g", "b", "is_start", "is_top", "n_faces", "flags",
]


def write_csv(doc: dict, path: Path) -> None:
    """
    Write one CSV row per hold, sorted by height.

    Args:
        doc (dict): Output document from extract().
        path (Path): Destination CSV path.

    Returns:
        None
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for r in sorted(doc["holds"], key=lambda h: h["height_m"]):
            facing = r["facing"] or [None] * 3
            color = r["mean_color"] or [None] * 3
            writer.writerow([
                r["id"], r["height_m"], *r["position"], r["wall_angle_deg"], r["wall_kind"],
                r["protrusion_m"], *r["footprint_m"], r["grip_area_m2"], r["slope_deg"],
                r["incut_angle_deg"], r["edge_depth_m"], r["two_hands"], *facing,
                *r["wall_normal"], *color, r["is_start"], r["is_top"], r["n_faces"],
                ";".join(r["flags"]),
            ])


def print_summary(doc: dict) -> None:
    """
    Print a table with the main features of each hold.

    Args:
        doc (dict): Output document from extract().

    Returns:
        None
    """

    def fmt(x: float | None, spec: str) -> str:
        """
        Format a possibly-missing number.

        Args:
            x (float | None): Value.
            spec (str): Format spec.

        Returns:
            str: Formatted value or "-".
        """
        return "-" if x is None else format(x, spec)

    header = f"{'id':<24}{'h(m)':>6}{'wall':>7} {'kind':<9}{'prot':>6}{'w x h (cm)':>12}{'grip cm2':>9}{'slope':>6}{'incut':>6}{'edge':>6} {'2H':<3} flags"
    print(header)
    print("-" * len(header))
    for r in doc["holds"]:
        w, h = r["footprint_m"]
        print(
            f"{r['id'][:23]:<24}{r['height_m']:>6.2f}{r['wall_angle_deg']:>7.1f} {r['wall_kind']:<9}"
            f"{r['protrusion_m'] * 100:>6.1f}{f'{w * 100:.0f} x {h * 100:.0f}':>12}{r['grip_area_m2'] * 1e4:>9.1f}"
            f"{fmt(r['slope_deg'], '.0f'):>6}{fmt(r['incut_angle_deg'], '+.0f'):>6}"
            f"{fmt(None if r['edge_depth_m'] is None else r['edge_depth_m'] * 100, '.1f'):>6} "
            f"{'Y' if r['two_hands'] else 'N':<3} {','.join(r['flags'])}"
        )
    print(f"start holds: {doc['start_hold_ids'] or 'none'}   top hold: {doc['top_hold_id'] or 'none'}")


def plot(doc: dict, context: dict, up_axis: str, out_path: str | None = None) -> None:
    """
    Show (or save) a 3D view of the wall around the route, holds, labels and arrows.

    Blue arrows are wall normals, red arrows are grip facings.

    Args:
        doc (dict): Output document from extract().
        context (dict): Context from extract() with the wall and hold meshes.
        up_axis (str): "y" or "z".
        out_path (str | None): Save the figure here instead of opening a window.

    Returns:
        None
    """
    import matplotlib

    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def to_display(p: np.ndarray) -> np.ndarray:
        """
        Map file coordinates to a right-handed Z-up display frame.

        Args:
            p (np.ndarray): Points (..., 3).

        Returns:
            np.ndarray: Display points (..., 3).
        """
        p = np.asarray(p, dtype=float)
        return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1) if up_axis == "y" else p

    wall, holds = context["wall"], context["holds"]
    hold_pts = np.vstack([h.mesh.vertices for h in holds])
    lo, hi = hold_pts.min(0) - 0.6, hold_pts.max(0) + 0.6
    centers = wall.mesh.triangles_center
    keep = np.all((centers > lo) & (centers < hi), axis=1)
    idx = np.flatnonzero(keep)
    if len(idx) > 25000:
        idx = np.random.default_rng(0).choice(idx, 25000, replace=False)

    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(to_display(wall.mesh.triangles[idx]), facecolor="0.55", edgecolor="none", alpha=0.35))
    records = {r["id"]: r for r in doc["holds"]}
    for h in holds:
        r = records[h.name]
        color = np.array(r["mean_color"]) / 255 if r["mean_color"] else (1.0, 0.5, 0.0)
        ax.add_collection3d(Poly3DCollection(to_display(h.mesh.triangles), facecolor=color, edgecolor="none"))
        p = np.array(r["position"])
        n = np.array(r["wall_normal"])
        ax.quiver(*to_display(p), *to_display(n) * 0.15, color="tab:blue")
        if r["facing"]:
            ax.quiver(*to_display(p), *to_display(np.array(r["facing"])) * 0.12, color="tab:red")
        ax.text(*to_display(p + n * 0.18), f"{r['id']}\n{r['wall_angle_deg']:+.0f} deg", fontsize=7)

    dlo, dhi = to_display(lo), to_display(hi)
    lo3, hi3 = np.minimum(dlo, dhi), np.maximum(dlo, dhi)
    ax.set_xlim(lo3[0], hi3[0]); ax.set_ylim(lo3[1], hi3[1]); ax.set_zlim(lo3[2], hi3[2])
    ax.set_box_aspect(hi3 - lo3)
    mean_n = to_display(np.mean([r["wall_normal"] for r in doc["holds"]], axis=0))
    ax.view_init(elev=15, azim=math.degrees(math.atan2(mean_n[1], mean_n[0])) + 40)
    ax.set_title(f"{doc['source_file']}  (blue: wall normal, red: grip facing)")
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"plot saved to {out_path}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser; every Params field becomes a --flag.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb", help="input GLB file")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--csv", help="CSV output path")
    parser.add_argument("--plot", action="store_true", help="show a 3D view")
    parser.add_argument("--plot-out", help="save the 3D view to this image file")
    for f in fields(Params):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "up":
            parser.add_argument(flag, choices=sorted(UP_VECTORS), default=f.default)
        elif f.name == "wall":
            parser.add_argument(flag, default=None)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point.

    Args:
        argv (list[str] | None): Arguments (defaults to sys.argv[1:]).

    Returns:
        None
    """
    args = build_parser().parse_args(argv)
    params = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})
    glb = Path(args.glb)
    out = Path(args.out) if args.out else glb.with_name(glb.stem + "_holds.json")
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")

    doc, context = extract(glb, params)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_csv(doc, csv_path)
    print_summary(doc)
    print(f"wrote {out} and {csv_path}")
    if args.plot or args.plot_out:
        plot(doc, context, params.up, args.plot_out)


if __name__ == "__main__":
    main()
