"""
Tests for extract_holds.py against a synthetic wall GLB with known answers.

Run: python -m pytest tests -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_holds  # noqa: E402
import synthetic_wall  # noqa: E402


def run(glb: Path, tmp_path: Path, *extra: str) -> dict:
    """
    Run the CLI on a GLB and load the JSON it writes.

    Args:
        glb (Path): Input GLB.
        tmp_path (Path): Directory for outputs.
        *extra (str): Additional command-line flags.

    Returns:
        dict: The parsed output document.
    """
    out = tmp_path / "holds.json"
    extract_holds.main([str(glb), "--out", str(out), *extra])
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """
    Build the default synthetic GLB once for the module.

    Args:
        tmp_path_factory (pytest.TempPathFactory): pytest temp dir factory.

    Returns:
        tuple[Path, dict]: GLB path and expected answers per hold.
    """
    path = tmp_path_factory.mktemp("synthetic") / "wall.glb"
    return path, synthetic_wall.build(path)


def check_holds(doc: dict, expected: dict) -> None:
    """
    Compare every extracted hold with its expected answers.

    Args:
        doc (dict): Output document.
        expected (dict): Expected answers per hold name.

    Returns:
        None
    """
    holds = {h["id"]: h for h in doc["holds"]}
    assert set(holds) == set(expected)
    up = extract_holds.UP_VECTORS[doc["up_axis"]]
    for name, exp in expected.items():
        h = holds[name]
        np.testing.assert_allclose(h["position"], exp["position"], atol=0.002, err_msg=name)
        assert h["height_m"] == pytest.approx(np.asarray(exp["position"]) @ up, abs=0.006), name
        assert h["wall_angle_deg"] == pytest.approx(exp["wall_angle_deg"], abs=1.0), name
        assert h["wall_kind"] == "overhang", name
        assert h["protrusion_m"] == pytest.approx(exp["protrusion_m"], abs=0.003), name
        assert h["incut_angle_deg"] == pytest.approx(exp["incut_angle_deg"], abs=2.0), name
        assert h["slope_deg"] == pytest.approx(exp["slope_deg"], abs=2.0), name
        assert h["edge_depth_m"] == pytest.approx(exp["edge_depth_m"], abs=0.003), name
        assert h["footprint_m"][0] == pytest.approx(exp["footprint_w"], abs=0.003), name
        assert h["two_hands"] == exp["two_hands"], name
        assert np.linalg.norm(h["wall_normal"]) == pytest.approx(1.0, abs=1e-4)
        assert "no_grip_surface" not in h["flags"], name
        assert "suspiciously_flat" not in h["flags"], name


def test_recovers_known_answers(synthetic: tuple[Path, dict], tmp_path: Path) -> None:
    """
    Positions, heights, angles, protrusion, incut/sloper and edge depth match the build.
    """
    glb, expected = synthetic
    doc = run(glb, tmp_path)
    assert doc["wall_object"] == "Wall_scan"
    assert abs(doc["wall_bottom_height"]) < 0.006
    check_holds(doc, expected)
    holds = {h["id"]: h for h in doc["holds"]}
    assert holds["jug_start"]["incut_angle_deg"] > 0
    assert holds["sloper_top"]["incut_angle_deg"] < 0
    assert holds["sloper_top"]["wall_angle_deg"] < holds["jug_start"]["wall_angle_deg"]


def test_names_colors_and_outputs(synthetic: tuple[Path, dict], tmp_path: Path) -> None:
    """
    start/top names, texture and vertex colors, missing colors, CSV ordering.
    """
    glb, _ = synthetic
    doc = run(glb, tmp_path)
    holds = {h["id"]: h for h in doc["holds"]}
    assert doc["start_hold_ids"] == ["jug_start"]
    assert doc["top_hold_id"] == "sloper_top"
    assert holds["jug_start"]["mean_color"] == pytest.approx([200, 30, 30], abs=2)
    assert holds["sloper_top"]["mean_color"] == pytest.approx([30, 180, 60], abs=2)
    assert holds["ledge"]["mean_color"] is None and "no_color" in holds["ledge"]["flags"]
    heights = [h["height_m"] for h in doc["holds"]]
    assert heights == sorted(heights)
    rows = (tmp_path / "holds.csv").read_text().splitlines()
    assert len(rows) == 1 + len(holds)
    assert rows[1].startswith("foot_chip")


def test_wall_fallback_to_largest(tmp_path: Path) -> None:
    """
    Without "wall" in any name, the largest-area object is used as the wall.
    """
    glb = tmp_path / "noname.glb"
    expected = synthetic_wall.build(glb, wall_name="scan_mesh")
    doc = run(glb, tmp_path)
    assert doc["wall_object"] == "scan_mesh"
    check_holds(doc, expected)


def test_z_up_file(tmp_path: Path) -> None:
    """
    A Z-up file processed with --up z gives the same angles and features.
    """
    glb = tmp_path / "zup.glb"
    expected = synthetic_wall.build(glb, up="z")
    doc = run(glb, tmp_path, "--up", "z")
    check_holds(doc, expected)


def test_inverted_scan_winding(tmp_path: Path) -> None:
    """
    A wall whose face normals all point into the wall still gives room-facing normals.
    """
    glb = tmp_path / "inverted.glb"
    expected = synthetic_wall.build(glb, invert_wall=True)
    doc = run(glb, tmp_path)
    check_holds(doc, expected)


def test_single_mesh_stops(tmp_path: Path) -> None:
    """
    A file with only one mesh (holds never separated) stops with instructions.
    """
    glb = tmp_path / "single.glb"
    trimesh.creation.box().export(str(glb))
    with pytest.raises(SystemExit, match="P > Selection"):
        extract_holds.main([str(glb), "--out", str(tmp_path / "x.json")])
