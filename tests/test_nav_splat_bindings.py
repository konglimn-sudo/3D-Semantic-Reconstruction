"""C++ bindings of the navigation map and the 3DGS support code."""

import numpy as np

from garage_recon import _core


def test_distance_transform_and_skeleton_graph():
    mask = np.zeros((40, 80), np.uint8)
    mask[10:17, 5:75] = 1   # Horizontal corridor ...
    mask[17:38, 37:44] = 1  # ... with a branch down.
    d = _core.distance_transform(1 - mask)
    assert d.shape == mask.shape and d[13, 20] == 4.0  # Rows 9 and 17 free.
    graph = _core.skeleton_graph(_core.thin_mask(mask), 6.0)
    assert len(graph["edges"]) == 3 and graph["nodes"].shape == (4, 2)
    simple = _core.simplify_polyline(graph["edges"][0]["cells"], 0.5)
    assert 2 <= len(simple) <= len(graph["edges"][0]["cells"])


def test_route_graph_and_grid_astar():
    graph = _core.RouteGraph(3)
    graph.add_edge(0, 1, 2.0)
    graph.add_edge(1, 2, 1.0, one_way=True)
    assert graph.shortest_path(0, 2)["nodes"] == [0, 1, 2]
    assert not graph.shortest_path(2, 0)["found"]
    assert np.isinf(graph.distances(2)[0])
    cost = np.zeros((10, 20), np.float32)
    cost[:8, 10] = -1.0  # A wall with a gap at the bottom rows.
    path = _core.grid_astar(cost, (2, 2), (17, 2))
    assert path[0].tolist() == [2, 2] and path[-1].tolist() == [17, 2]
    assert path[path[:, 0] == 10, 1].min() >= 8


def test_gaussians_render_a_wall_along_rays():
    y, z = np.meshgrid(np.arange(-30, 31) * 0.05, np.arange(-30, 31) * 0.05)
    points = np.column_stack([np.full(y.size, 4.0), y.ravel(), z.ravel()])
    normals = np.tile([-1.0, 0.0, 0.0], (len(points), 1))
    init = _core.init_gaussians(points, normals, np.full(len(points), 0.002))
    assert init["quats"].shape == (len(points), 4)
    renderer = _core.GaussianRayRenderer(
        points, init["quats"], init["scales"], init["opacity"],
        intensity=np.full(len(points), 0.7))
    directions = np.array([[1.0, 0.1, 0.05], [1.0, -0.2, 0.0]])
    out = renderer.render(np.zeros((2, 3)), directions)
    expected = 4.0 * np.linalg.norm(directions, axis=1)
    assert np.allclose(out["depth"], expected, atol=0.005)
    assert np.allclose(out["intensity"], 0.7, atol=1e-3)
    assert (out["opacity"] > 0.99).all()


def test_camera_coverage_and_training_mask():
    camera = _core.make_camera("pinhole", 100, 80, [50, 50, 50, 40])
    x, y = np.meshgrid(np.arange(-40, 41) * 0.05, np.arange(-30, 31) * 0.05)
    wall = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 3.0)])
    coverage = _core.CameraCoverage(wall)
    image = np.full((80, 100, 3), 120, np.uint8)
    image[:, 50:] = 2  # Right half dark.
    coverage.add_frame(camera, np.eye(4), image)
    result = coverage.result()
    left = np.argmin(np.linalg.norm(wall - [-1.0, 0.0, 3.0], axis=1))
    right = np.argmin(np.linalg.norm(wall - [1.0, 0.0, 3.0], axis=1))
    assert result["good_views"][left] == 1 and result["good_views"][right] == 0
    assert result["views"][right] == 1
    assert np.allclose(result["color"][left], 120 / 255, atol=1e-6)
    index = _core.PointCloudIndex(wall)
    moving = np.array([[0.0, 0.0, 2.0]])
    mask = _core.render_training_mask(camera, np.eye(4), index, moving=moving)
    assert mask.shape == (80, 100) and mask[40, 50] == _core.MASK_MOVING
    assert mask[5, 5] == 0


def test_free_space_mask():
    carver = _core.FreeSpaceCarver([-1, -1, -1], [11, 1, 1], 0.2)
    ends = np.array([[10.0, 0.0, 0.0]])
    for _ in range(3):
        carver.integrate([0.0, 0.0, 0.0], ends)
    free = carver.free_mask(2, 0)
    assert free.shape == tuple(carver.dims)
    def cell(p):
        offset = (np.asarray(p) - carver.min_bound) / carver.voxel
        return tuple(offset.astype(int))
    assert free[cell([5.0, 0.05, 0.05])] == 1
    assert free[cell([10.0, 0.05, 0.05])] == 0
