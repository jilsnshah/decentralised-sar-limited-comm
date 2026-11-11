"""An agent's (partial) knowledge of the floor plan.

Cells are UNKNOWN until they come within an agent's sensing radius (or are
reported by a teammate). Path planning is optimistic: unknown cells are
assumed traversable, and plans are recomputed every step as cells are
revealed, so an agent "bumps" into a wall it did not know about and re-routes.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .env import FREE, HAZARD, OBSTACLE

UNKNOWN = -1


class MapKnowledge:
    def __init__(self, building, known=False):
        self.n = building.n
        if known:
            self.grid = building.grid.astype(np.int8).copy()
        else:
            self.grid = np.full((self.n, self.n), UNKNOWN, dtype=np.int8)
            self.grid[building.exit] = FREE
        self.version = 0
        self._cache = {}

    # ------------------------------------------------------------ updates
    def reveal(self, ys, xs, truth):
        """Observe the true terrain of cells (ys, xs). Returns newly learnt
        cells as a list of (y, x, value)."""
        cur = self.grid[ys, xs]
        new = cur == UNKNOWN
        if not new.any():
            return []
        ny, nx = ys[new], xs[new]
        vals = truth[ny, nx]
        self.grid[ny, nx] = vals
        self.version += 1
        return list(zip(ny.tolist(), nx.tolist(), vals.tolist()))

    def merge(self, cells):
        """Merge terrain reported by a teammate."""
        changed = False
        for y, x, v in cells:
            if self.grid[y, x] == UNKNOWN:
                self.grid[y, x] = v
                changed = True
        if changed:
            self.version += 1

    def blocked_for_victim(self, cells):
        """Cells in `cells` where the victim cannot be (obstacle or hazard)."""
        return [(y, x) for y, x, v in cells if v != FREE]

    # ------------------------------------------------------------ planning
    def passable(self, kind):
        if kind == "drone":
            return self.grid != OBSTACLE
        return (self.grid == FREE) | (self.grid == UNKNOWN)

    def plan(self, pos, kind):
        """Shortest-path distances (n x n array) and predecessor array from
        `pos` on the believed map; cached until the map changes."""
        key = (pos, kind)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == self.version:
            return hit[1], hit[2]
        n = self.n
        ok = self.passable(kind)
        ok[pos] = True
        idx = np.arange(n * n).reshape(n, n)
        rows, cols = [], []
        for dy, dx in ((0, 1), (1, 0)):
            a = ok[: n - dy, : n - dx] & ok[dy:, dx:]
            u = idx[: n - dy, : n - dx][a]
            v = idx[dy:, dx:][a]
            rows += [u, v]
            cols += [v, u]
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n))
        d, pred = dijkstra(g, directed=True, indices=pos[0] * n + pos[1],
                           unweighted=True, return_predecessors=True)
        D = d.reshape(n, n)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = (self.version, D, pred)
        return D, pred

    def path(self, pos, target, kind):
        """List of cells from pos (exclusive) to target (inclusive); [] if
        the target is believed unreachable or equal to pos."""
        if pos == target:
            return []
        D, pred = self.plan(pos, kind)
        if not np.isfinite(D[target]):
            return []
        n = self.n
        src = pos[0] * n + pos[1]
        k = target[0] * n + target[1]
        out = []
        while k != src and k >= 0:
            out.append(divmod(int(k), n))
            k = pred[k]
        out.reverse()
        return out


def truly_passable(building, cell, kind):
    v = building.grid[cell]
    if kind == "drone":
        return v != OBSTACLE
    return v == FREE


__all__ = ["MapKnowledge", "UNKNOWN", "truly_passable", "FREE", "HAZARD", "OBSTACLE"]
