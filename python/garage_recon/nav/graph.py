"""Route graph of the drivable aisles, connected across levels by the ramps.

The aisle graph of a level is the skeleton (C++ thinning and graph
extraction) of the drivable cells outside slots and ramps that keep half the
vehicle width plus a margin from every obstacle or unknown cell. Slots are
not part of the aisles, so routes run between the slot rows. Every point
of an aisle edge therefore has that clearance, and a route along the graph
cannot pass through a wall or a column. Each ramp becomes an edge from its
top to its bottom, joined to the aisle graphs of both levels by straight
connectors that are checked against the obstacle layer. Slots join the graph
the same way (see goals.py).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from garage_recon import _core


@dataclasses.dataclass
class Edge:
    """A graph edge.

    Attributes:
        a: First node.
        b: Last node.
        points: (M, 3) polyline from node a to node b (world).
        kind: "aisle", "lane" (from a row of slots, lanes.py), "ramp",
            "connector" or "bridge" (joins pieces of an aisle graph through
            tight or slot cells; for review).
        level: Level of the edge (the upper level of a ramp).
        width: Free width along the edge (m): twice the smallest clearance.
        one_way: Travelled from a to b only (floor arrows, traffic.py).
        headroom: Lowest object above the floor along the edge (m; None
            for open sky or ramps; attributes.py).
        posted_height: Posted height limit (m) of a sign along the edge.
        features: Road features on the edge ("barrier": a barrier gate,
            "speed_bump"; attributes.py).
        blocked: A later trip found it impassable (update.py); routes
            avoid it.
    """

    a: int
    b: int
    points: np.ndarray
    kind: str
    level: int
    width: float
    one_way: bool = False
    headroom: float | None = None
    posted_height: float | None = None
    features: tuple = ()
    blocked: bool = False

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def to_json(self) -> dict:
        return {"a": self.a, "b": self.b, "kind": self.kind,
                "level": self.level, "width": round(self.width, 3),
                "one_way": self.one_way, "length": round(self.length, 3),
                "headroom": self.headroom,
                "posted_height": self.posted_height,
                "features": list(self.features),
                "blocked": self.blocked,
                "points": np.round(self.points, 3).tolist()}

    @property
    def height_limit(self) -> float | None:
        """The lower of the measured headroom and a posted limit."""
        values = [v for v in (self.headroom, self.posted_height)
                  if v is not None]
        return min(values) if values else None


@dataclasses.dataclass
class Attachment:
    """Nearest point of an edge to a query point."""

    distance: float
    edge: int
    s: float  # Arc length from the edge's first point (m, in xy).
    point: np.ndarray  # (3,)
    tangent: np.ndarray  # (2,) unit, along the edge from a to b.


@dataclasses.dataclass
class NavGraph:
    """Nodes (xyz, level) and edges of all levels."""

    nodes: list[np.ndarray] = dataclasses.field(default_factory=list)
    node_level: list[int] = dataclasses.field(default_factory=list)
    edges: list[Edge] = dataclasses.field(default_factory=list)

    def add_node(self, xyz, level: int) -> int:
        self.nodes.append(np.asarray(xyz, float).reshape(3))
        self.node_level.append(int(level))
        return len(self.nodes) - 1

    def add_edge(self, edge: Edge) -> int:
        self.edges.append(edge)
        return len(self.edges) - 1

    # ------------------------------------------------------------ queries
    def nearest(self, xy, level: int, kinds=("aisle", "lane", "bridge"),
                direction=None, max_distance: float = np.inf
                ) -> list[Attachment]:
        """Projections of `xy` on the edges of a level, nearest first.

        With `direction` (unit xy) only edges that the ray xy + t direction
        (t > 0) crosses are returned, ordered by t.
        """
        xy = np.asarray(xy, float)[:2]
        out = []
        for index, edge in enumerate(self.edges):
            if (edge.level != level or edge.kind not in kinds
                    or len(edge.points) < 2):
                continue
            p = edge.points[:, :2]
            seg = np.diff(p, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            if direction is None:
                t = np.clip(np.einsum("ij,ij->i", xy - p[:-1], seg)
                            / np.maximum(seg_len ** 2, 1e-12), 0.0, 1.0)
                foot = p[:-1] + t[:, None] * seg
                distance = np.linalg.norm(foot - xy, axis=1)
                k = int(np.argmin(distance))
                key = float(distance[k])
            else:
                d = np.asarray(direction, float)[:2]
                # Solve xy + r d = p0 + t seg.
                det = seg[:, 0] * (-d[1]) - seg[:, 1] * (-d[0])
                ok = np.abs(det) > 1e-9
                rel = xy - p[:-1]
                t = np.where(ok, (rel[:, 0] * (-d[1]) - rel[:, 1] * (-d[0]))
                             / np.where(ok, det, 1.0), -1.0)
                r = np.where(ok, (seg[:, 0] * rel[:, 1] - seg[:, 1] * rel[:, 0])
                             / np.where(ok, det, 1.0), -1.0)
                hit = ok & (t >= 0.0) & (t <= 1.0) & (r > 0.0)
                if not hit.any():
                    continue
                r = np.where(hit, r, np.inf)
                k = int(np.argmin(r))
                key = float(r[k])
            if key > max_distance:
                continue
            s = cum[k] + t[k] * seg_len[k]
            point = edge.points[k] + t[k] * (edge.points[k + 1]
                                             - edge.points[k])
            tangent = seg[k] / max(seg_len[k], 1e-12)
            out.append(Attachment(key, index, float(s), point, tangent))
        out.sort(key=lambda a: a.distance)
        return out

    def split(self, edge_index: int, s: float, snap: float = 0.3) -> int:
        """Node at arc length `s` of an edge, splitting the edge if needed
        (existing end nodes within `snap` metres are reused)."""
        edge = self.edges[edge_index]
        p = edge.points
        seg_len = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        if s <= snap:
            return edge.a
        if s >= cum[-1] - snap:
            return edge.b
        k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg_len) - 1))
        t = (s - cum[k]) / max(seg_len[k], 1e-12)
        point = p[k] + t * (p[k + 1] - p[k])
        node = self.add_node(point, self.node_level[edge.a])
        tail = dataclasses.replace(edge, a=node,
                                   points=np.vstack([point, p[k + 1:]]))
        edge.points = np.vstack([p[:k + 1], point])
        edge.b = node
        self.add_edge(tail)
        return node

    def route_graph(self, reverse: bool = False,
                    two_way: bool = False) -> _core.RouteGraph:
        """C++ route graph; `reverse` turns every one-way edge around
        (distances *to* a node), `two_way` ignores the one-way edges'
        direction."""
        graph = _core.RouteGraph(len(self.nodes))
        for edge in self.edges:
            a, b = (edge.b, edge.a) if reverse else (edge.a, edge.b)
            if edge.blocked:  # A loop keeps the edge numbering.
                graph.add_edge(a, a, 0.0, True)
                continue
            graph.add_edge(a, b, edge.length, edge.one_way and not two_way)
        return graph

    def path_points(self, nodes: list[int], edges: list[int]) -> np.ndarray:
        """(M, 3) polyline of a path given as nodes and edges."""
        parts = []
        for k, e in enumerate(edges):
            edge = self.edges[e]
            points = edge.points if edge.a == nodes[k] else edge.points[::-1]
            parts.append(points if not parts else points[1:])
        if not parts:
            return np.asarray([self.nodes[nodes[0]]]) if nodes else np.zeros(
                (0, 3))
        return np.vstack(parts)

    # --------------------------------------------------------------- I/O
    def to_json(self) -> dict:
        return {"nodes": [{"id": i, "level": lv,
                           "xyz": np.round(p, 3).tolist()}
                          for i, (p, lv) in enumerate(zip(self.nodes,
                                                          self.node_level))],
                "edges": [e.to_json() for e in self.edges]}

    @classmethod
    def from_json(cls, data: dict) -> NavGraph:
        """The graph written by to_json."""
        graph = cls()
        for node in data["nodes"]:
            graph.add_node(node["xyz"], node["level"])
        for e in data["edges"]:
            graph.add_edge(Edge(e["a"], e["b"], np.asarray(e["points"], float),
                                e["kind"], e["level"], e["width"],
                                e["one_way"], e.get("headroom"),
                                e.get("posted_height"),
                                tuple(e.get("features", ())),
                                bool(e.get("blocked", False))))
        return graph


def segment_free(layers, p, q, step: float | None = None,
                 ignore_parked: bool = False) -> tuple[bool, float]:
    """Whether the straight segment p-q (world xy) avoids structure (and
    parked) cells; also the smallest clearance along it (m)."""
    p, q = np.asarray(p, float)[:2], np.asarray(q, float)[:2]
    step = step or 0.5 * layers.grid.res
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / step)) + 1)
    xy = p + np.linspace(0.0, 1.0, n)[:, None] * (q - p)
    blocked = layers.sample("structure", xy, True)
    if not ignore_parked:
        blocked = blocked | layers.sample("parked", xy, True)
    clearance = layers.sample("clearance", xy, 0.0)
    return not bool(blocked.any()), float(np.min(clearance))


def add_level_graph(graph: NavGraph, layers, cfg) -> int:
    """Adds the aisle skeleton of one level; returns the number of edges."""
    nav = cfg.nav
    grid = layers.grid
    half = 0.5 * nav.vehicle.width + nav.vehicle.margin
    # Off the slots, unless the survey vehicle drove there (a slot detected
    # on an aisle, or an empty slot crossed): those cells are drivable.
    off_slots = layers.slot == 0
    if layers.driven is not None:
        off_slots = off_slots | layers.driven
    aisle = layers.drivable & off_slots & ~layers.ramp
    navigable = aisle & (layers.clearance >= half)
    skeleton = _core.thin_mask(navigable.astype(np.uint8))
    result = _core.skeleton_graph(skeleton, nav.min_spur / grid.res)

    def world(cells: np.ndarray) -> np.ndarray:
        col = np.clip(np.round(cells[:, 0]).astype(int), 0, grid.width - 1)
        row = np.clip(np.round(cells[:, 1]).astype(int), 0, grid.height - 1)
        z = layers.floor_z[row, col]
        z = np.where(np.isfinite(z), z, np.nanmedian(layers.floor_z))
        return np.column_stack([grid.x0 + (cells[:, 0] + 0.5) * grid.res,
                                grid.y0 + (cells[:, 1] + 0.5) * grid.res, z])

    first = len(graph.nodes)
    for xyz in world(np.asarray(result["nodes"]).reshape(-1, 2)):
        graph.add_node(xyz, grid.level)
    for e in result["edges"]:
        cells = np.asarray(e["cells"])
        col = np.clip(np.round(cells[:, 0]).astype(int), 0, grid.width - 1)
        row = np.clip(np.round(cells[:, 1]).astype(int), 0, grid.height - 1)
        width = 2.0 * float(np.min(layers.clearance[row, col]))
        simple = _core.simplify_polyline(np.ascontiguousarray(cells, float),
                                         nav.simplify / grid.res)
        points = world(simple)
        points[0] = graph.nodes[first + e["a"]]
        points[-1] = graph.nodes[first + e["b"]]
        graph.add_edge(Edge(first + e["a"], first + e["b"], points, "aisle",
                            grid.level, width))
    return len(result["edges"])


@dataclasses.dataclass
class Connection:
    """A point joined to the aisle graph.

    Attributes:
        node: The point's own node (the aisle node for a zero-length
            connector).
        access_node: Node on the aisle centre line.
        length: Connector length (m).
        tangent: (2,) unit direction of the aisle at the access node.
    """

    node: int
    access_node: int
    length: float
    tangent: np.ndarray


def connect(graph: NavGraph, layers, xyz, level: int, max_distance: float,
            direction=None, ignore_parked: bool = False) -> Connection | None:
    """Joins a point to the aisle graph of its level with a straight
    connector that avoids the obstacles.

    With `direction` the aisle centre line is searched along that ray first,
    then by plain distance.
    """
    xyz = np.asarray(xyz, float)
    candidates = []
    if direction is not None:
        candidates += graph.nearest(xyz[:2], level, direction=direction,
                                    max_distance=max_distance)
    candidates += graph.nearest(xyz[:2], level, max_distance=max_distance)
    for attachment in candidates:
        free, _ = segment_free(layers, xyz, attachment.point,
                               ignore_parked=ignore_parked)
        if not free:
            continue
        # The split may snap to an end node: check the connector it leaves.
        access = graph.split(attachment.edge, attachment.s)
        target = graph.nodes[access]
        free, clearance = segment_free(layers, xyz, target,
                                       ignore_parked=ignore_parked)
        if not free:
            continue
        length = float(np.linalg.norm(target[:2] - xyz[:2]))
        node = access
        if length > 1e-6:
            node = graph.add_node(xyz, level)
            graph.add_edge(Edge(node, access, np.vstack([xyz, target]),
                                "connector", level, 2.0 * clearance))
        return Connection(node, access, length, attachment.tangent)
    return None


def ramp_ends(ramp) -> tuple[np.ndarray, np.ndarray, float]:
    """(top, bottom, width): ends of a ramp's centre line along its slope
    (xyz) and its width across the slope (m)."""
    gradient = np.array(ramp.plane[:2], float)  # A copy: normalised below.
    norm = np.linalg.norm(gradient)
    if norm < 1e-6:
        raise ValueError("flat ramp")
    gradient /= norm
    polygon = np.asarray(ramp.polygon, float)
    center = polygon.mean(axis=0)
    s = (polygon - center) @ gradient
    across = (polygon - center) @ np.array([-gradient[1], gradient[0]])
    ends = []
    for value in (s.max(), s.min()):
        xy = center + value * gradient
        ends.append(np.array([xy[0], xy[1], float(ramp.z_at(xy[None])[0])]))
    return ends[0], ends[1], float(np.ptp(across))


def _components(graph: NavGraph) -> list[int]:
    """Component label of every node (union-find over all edges)."""
    parent = list(range(len(graph.nodes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for edge in graph.edges:
        parent[find(edge.a)] = find(edge.b)
    return [find(a) for a in range(len(graph.nodes))]


def bridge_components(graph: NavGraph, layers, cfg) -> int:
    """Joins the pieces of a level's aisle graph with grid A* paths.

    A misdetected slot or a pinch point below the comfortable clearance cuts
    the aisle skeleton. A bridge may cross slots (at a cost) and only needs
    the vehicle's half width without the margin; bridges are edges of kind
    "bridge" for review. Pieces that no such path joins stay apart (a real
    pinch point). Returns the number of bridges.
    """
    nav = cfg.nav
    grid = layers.grid
    level = grid.level
    half = 0.5 * nav.vehicle.width
    comfortable = half + nav.vehicle.margin
    passable = layers.drivable & (layers.clearance >= half)
    cost = np.where(passable, 0.0, -1.0).astype(np.float32)
    cost[passable & (layers.clearance < comfortable)] = nav.bridge_tight_cost
    cost[passable & (layers.slot > 0)] += nav.bridge_slot_cost
    used = {n for e in graph.edges for n in (e.a, e.b)}
    nodes = [k for k, lv in enumerate(graph.node_level)
             if lv == level and k in used]  # Not nodes of dropped edges.
    if len(nodes) < 2:
        return 0
    xy = np.asarray([graph.nodes[k][:2] for k in nodes])
    col, row, inside = layers.cells(xy)
    bridges = 0
    tried = set()
    while True:
        labels = _components(graph)
        comp = np.asarray([labels[k] for k in nodes])
        if len(set(comp)) < 2:
            return bridges
        sizes = {c: 0.0 for c in set(comp)}
        for edge in graph.edges:
            if edge.level == level and labels[edge.a] in sizes:
                sizes[labels[edge.a]] += edge.length
        main = max(sizes, key=sizes.get)
        a_idx = np.flatnonzero((comp == main) & inside)
        best = None
        for other in sorted(set(comp) - {main}):
            if (main, other) in tried:
                continue
            b_idx = np.flatnonzero((comp == other) & inside)
            if not len(a_idx) or not len(b_idx):
                tried.add((main, other))
                continue
            d = np.linalg.norm(xy[a_idx][:, None] - xy[b_idx][None], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] <= nav.max_bridge and (best is None
                                              or d[i, j] < best[0]):
                best = (d[i, j], other, a_idx[i], b_idx[j])
        if best is None:
            return bridges
        _, other, i, j = best
        tried.add((main, other))
        path = _core.grid_astar(cost, (int(col[i]), int(row[i])),
                                (int(col[j]), int(row[j])))
        if len(path) == 0:
            continue
        cells = _core.simplify_polyline(path.astype(float),
                                        nav.simplify / grid.res)
        z = layers.floor_z[cells[:, 1].astype(int), cells[:, 0].astype(int)]
        points = np.column_stack([grid.x0 + (cells[:, 0] + 0.5) * grid.res,
                                  grid.y0 + (cells[:, 1] + 0.5) * grid.res,
                                  np.where(np.isfinite(z), z,
                                           graph.nodes[nodes[i]][2])])
        if len(points) < 2:  # Both nodes in one cell.
            points = np.vstack([points, points])
        points[0], points[-1] = graph.nodes[nodes[i]], graph.nodes[nodes[j]]
        width = 2.0 * float(layers.clearance[path[:, 1], path[:, 0]].min())
        graph.add_edge(Edge(nodes[i], nodes[j], points, "bridge", level, width))
        bridges += 1


def build_graph(layers_by_level: dict, levels, cfg, log=print,
                slots=(), walls=(), drives=()) -> tuple[NavGraph, dict]:
    """Aisle graphs of all levels (with the lanes of the slot rows, walls
    and drives in open areas, see lanes.py) joined by the ramps; also the
    lane report per level."""
    # pylint: disable-next=import-outside-toplevel
    from garage_recon.nav import lanes as lanes_lib  # Imports this module.
    nav = cfg.nav
    graph = NavGraph()
    lane_reports = {}
    for level, layers in sorted(layers_by_level.items()):
        count = add_level_graph(graph, layers, cfg)
        lanes = lanes_lib.add_level_lanes(graph, layers, slots, cfg, walls,
                                          drives)
        lane_reports[level] = lanes
        bridges = bridge_components(graph, layers, cfg)
        log(f"[nav] level {level}: {count} aisle edges, {lanes['lanes']} "
            f"lanes from {lanes['rows']} slot rows, "
            f"{lanes.get('walls', 0)} walls and {lanes.get('drives', 0)} "
            f"drive stretches ({lanes['lane_length_m']} "
            f"m, {lanes['replaced_edges']} skeleton edges replaced, "
            f"{lanes['links']} links, {lanes.get('pruned_edges', 0)} "
            f"fragment edges dropped), {bridges} bridges")
    for ramp in levels.ramps:
        try:
            top, bottom, width = ramp_ends(ramp)
        except ValueError:
            continue
        ends = []
        for xyz, level in ((top, ramp.level_from), (bottom, ramp.level_to)):
            layers = layers_by_level.get(level)
            ends.append(None if layers is None else connect(
                graph, layers, xyz, level, nav.max_ramp_attach))
        if None in ends:
            log(f"[nav] ramp {ramp.level_from}->{ramp.level_to}: an end does "
                f"not reach the aisles")
            continue
        t = np.linspace(0.0, 1.0, 8)[:, None]
        points = top + t * (bottom - top)
        points[:, 2] = ramp.z_at(points[:, :2])
        graph.add_edge(Edge(ends[0].node, ends[1].node, points, "ramp",
                            ramp.level_from, width))
        log(f"[nav] ramp {ramp.level_from}->{ramp.level_to}: "
            f"{np.linalg.norm(bottom - top):.1f} m, width {width:.1f} m")
    return graph, lane_reports
