"""Unit tests of the section-5 building blocks on hand-made inputs."""

import numpy as np

from garage_recon.config import load_config
from garage_recon.recon.bev import BevGrid
from garage_recon.semantics.markings import MarkingSegment, extract_markings
from garage_recon.semantics.objects4d import Box3D
from garage_recon.semantics.rules import fix_slot_numbers
from garage_recon.semantics.slots import assemble_slots
from garage_recon.semantics.structure import Column

CFG = load_config()


def _row_segments(xs, y0, y1, level=0):
    return [MarkingSegment(level, [x, y0], [x, y1], 0.15, "parking_line", 1.0) for x in xs]


def _assemble(segs, path, columns=(), grid=None):
    grid = grid or BevGrid(0, -5.0, -5.0, 0.1, 400, 400)
    hm = np.zeros((grid.height, grid.width))
    fz = np.zeros((grid.height, grid.width))
    return assemble_slots(segs, 0, 0.0, path, list(columns), grid, hm, fz, CFG, log=lambda *_: None)


def test_slots_from_separators_entrance_faces_aisle():
    # Row of 3 bays (y 0..5.3) with the aisle (driven path) at y = 8.
    path = np.column_stack([np.linspace(-3, 12, 50), np.full(50, 8.0)])
    slots = _assemble(_row_segments([0.0, 2.5, 5.0, 7.5], 0.0, 5.3), path)
    assert len(slots) == 3
    for s in slots:
        c = np.asarray(s.corners)
        assert np.allclose(c[[0, 1], 1], 5.3)  # entrance corners on the aisle side
        assert abs(s.width - 2.5) < 1e-6 and abs(s.depth - 5.3) < 1e-6
        assert not s.inferred


def test_hidden_separator_is_inferred():
    path = np.column_stack([np.linspace(-3, 12, 50), np.full(50, 8.0)])
    slots = _assemble(_row_segments([0.0, 2.5, 7.5], 0.0, 5.3), path)  # x = 5.0 hidden by a car
    assert len(slots) == 3
    assert sum(s.inferred for s in slots) == 2


def test_column_faces_bound_bays():
    path = np.column_stack([np.linspace(-3, 20, 50), np.full(50, 8.0)])
    # Separator at the group boundary (x = 7.75) is hidden; the column at x = 8 bounds the bay instead.
    segs = _row_segments([0.25, 2.75, 5.25, 10.75, 13.25], 0.0, 5.3)
    col = Column("c", 0, [8.0, 5.05], [0.5, 0.5], 0.0, 0.0, 3.3, False, "A2")
    slots = _assemble(segs, path, [col])
    xs = sorted(round(float(np.mean(np.asarray(s.corners)[:, 0])), 2) for s in slots)
    assert 6.5 in xs and 9.5 in xs  # bays on both sides of the column


def test_markings_from_intensity_raster():
    grid = BevGrid(0, 0.0, 0.0, 0.05, 200, 140)
    inten = np.full((140, 200), 0.2)
    for x in (2.0, 4.5, 7.0):
        inten[10:120, int(x / 0.05) - 1:int(x / 0.05) + 2] = 0.85  # 15 cm wide, 5.5 m long
    rasters = {"intensity": inten, "height_max": np.full((140, 200), np.nan)}
    segs, _ = extract_markings(rasters, grid, CFG, yaw=0.0, log=lambda *_: None)
    assert len(segs) == 3
    for s in segs:
        assert 5.0 < s.length < 5.8
        assert 0.08 < s.width < 0.3


def test_box_contains():
    b = Box3D(1, "vehicle", [0.0, 0.0, 0.75], [4.0, 2.0, 1.5], np.pi / 2, True, [0, 0, 0], 10)
    pts = np.array([[0.0, 1.9, 0.5], [1.9, 0.0, 0.5], [0.0, 0.0, 2.0]])
    assert b.contains(pts).tolist() == [True, False, False]


def test_fix_slot_numbers_repairs_ocr():
    # "0I3" is a misread 013 (only digit runs that span the whole token are trusted)
    fixed = fix_slot_numbers([None] * 5, ["011", "012", "0I3", "014", None])
    assert fixed == ["011", "012", "013", "014", "015"]
    assert fix_slot_numbers([None] * 3, ["20", "19", None]) == ["20", "19", "18"]  # descending rows
