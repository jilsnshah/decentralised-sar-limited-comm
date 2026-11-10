"""Hazardous-building environment: a grid graph G=(V,E) with obstacles,
hazards, one exit and one (hidden) victim location."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

FREE, OBSTACLE, HAZARD = 0, 1, 2
MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _apsp(passable: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path (hop) distances on the 4-connected grid
    restricted to passable cells. Unreachable pairs get +inf."""
    H, W = passable.shape
    idx = np.arange(H * W).reshape(H, W)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = passable[: H - dy, : W - dx] & passable[dy:, dx:]
        u = idx[: H - dy, : W - dx][a]
        v = idx[dy:, dx:][a]
        rows += [u, v]
        cols += [v, u]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(H * W, H * W))
    d = shortest_path(g, method="D", unweighted=True, directed=False)
    d[~passable.ravel(), :] = np.inf
    d[:, ~passable.ravel()] = np.inf
    np.fill_diagonal(d, np.where(passable.ravel(), 0.0, np.inf))
    return d


class Building:
    """Randomly generated building. Drones fly over hazards but not through
    walls/obstacles; ground robots may not enter hazards."""

    def __init__(self, cfg, seed: int):
        rng = np.random.default_rng(seed)
        n = cfg.size
        self.n = n
        for _ in range(500):
            grid = np.zeros((n, n), dtype=np.int8)
            # internal walls with door gaps -> room structure
            for w in range(cfg.n_walls):
                vertical = rng.random() < 0.5
                pos = rng.integers(3, n - 3)
                lo = rng.integers(0, n // 3)
                hi = rng.integers(2 * n // 3, n + 1)
                line = np.arange(lo, hi)
                n_doors = 1 + (hi - lo) // 8
                doors = rng.choice(line, size=min(n_doors, len(line)), replace=False)
                keep = np.setdiff1d(line, np.concatenate([doors, doors + 1]))
                if vertical:
                    grid[keep, pos] = OBSTACLE
                else:
                    grid[pos, keep] = OBSTACLE
            # scattered debris
            free = np.argwhere(grid == FREE)
            k = int(cfg.obstacle_density * n * n)
            pick = free[rng.choice(len(free), size=k, replace=False)]
            grid[pick[:, 0], pick[:, 1]] = OBSTACLE
            # hazards (fire, collapse zones)
            free = np.argwhere(grid == FREE)
            k = int(cfg.hazard_density * n * n)
            pick = free[rng.choice(len(free), size=k, replace=False)]
            grid[pick[:, 0], pick[:, 1]] = HAZARD
            # exit on the left wall
            ey = int(rng.integers(1, n - 1))
            grid[ey, 0] = FREE
            if grid[ey, 1] == OBSTACLE:
                grid[ey, 1] = FREE
            self.exit = (ey, 0)
            self.grid = grid
            self.drone_ok = grid != OBSTACLE
            self.robot_ok = grid == FREE
            self.dist = {"drone": _apsp(self.drone_ok), "robot": _apsp(self.robot_ok)}
            ex = self.cid(self.exit)
            reach = np.isfinite(self.dist["robot"][ex]).reshape(n, n)
            # require most of the free space to be robot-reachable
            if reach.sum() < 0.6 * self.robot_ok.sum():
                continue
            self.robot_reach = reach
            far = np.argwhere(reach & (self.dist["robot"][ex].reshape(n, n) >= n * 0.6))
            if len(far) == 0:
                continue
            v = far[rng.integers(len(far))]
            self.victim = (int(v[0]), int(v[1]))
            break
        else:  # pragma: no cover
            raise RuntimeError("could not generate a building")
        # prior support for the victim: robot-reachable non-hazard cells
        self.prior_mask = self.robot_reach.copy()
        # start cells: all agents enter through the exit area
        near = np.argwhere(reach)
        dd = self.dist["robot"][ex][near[:, 0] * n + near[:, 1]]
        order = np.argsort(dd, kind="stable")
        self.start_cells = [tuple(map(int, near[i])) for i in order[: cfg.n_agents + 4]]
        # search regions
        ry, rx = cfg.regions
        ys = np.linspace(0, n, ry + 1).astype(int)
        xs = np.linspace(0, n, rx + 1).astype(int)
        self.region_of = np.zeros((n, n), dtype=int)
        self.region_cells = []
        k = 0
        for i in range(ry):
            for j in range(rx):
                self.region_of[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = k
                cells = np.zeros((n, n), dtype=bool)
                cells[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
                self.region_cells.append(cells)
                k += 1
        self.n_regions = k

    def cid(self, p) -> int:
        return p[0] * self.n + p[1]

    def neighbours(self, p, kind):
        ok = self.drone_ok if kind == "drone" else self.robot_ok
        y, x = p
        for dy, dx in MOVES:
            q = (y + dy, x + dx)
            if 0 <= q[0] < self.n and 0 <= q[1] < self.n and ok[q]:
                yield q

    def step_towards(self, p, target, kind):
        """One move along a shortest path from p to target."""
        if p == target:
            return p
        D = self.dist[kind]
        t = self.cid(target)
        best, bd = p, D[t, self.cid(p)]
        for q in self.neighbours(p, kind):
            dq = D[t, self.cid(q)]
            if dq < bd:
                best, bd = q, dq
        return best
