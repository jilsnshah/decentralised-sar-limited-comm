"""Task allocation: the CSP/DCOP of Section 3.

Tasks T = {SEARCH_k : k regions} U {RESCUE} U {IDLE}.  Capability matrix F
(Eq. 3-4): drones may not take RESCUE; IDLE is always feasible and is exempt
from the at-most-one constraint (2) (it is the "unassigned" option).

Both coordinators minimise the same cost  sum_i u_i(x_i) + sum_{i<j} w(x_i,x_j):
  * centralised  -> exact assignment with the Hungarian method on pooled info,
  * decentralised-> MGM (Maximum Gain Message) local search over the
                    communication graph, each agent using only its own belief.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

IDLE = ("idle",)
RESCUE = ("rescue",)
BIG = 1e6


def search(k):
    return ("search", k)


def feasible(agent, task) -> bool:
    """F_ij (Eq. 3-4)."""
    if task[0] == "rescue":
        return not agent.is_drone
    return True


def region_masses(building, prob):
    return np.array([prob[m].sum() for m in building.region_cells])


def unary_costs(agent, building, cfg, prob, claim, row):
    """u_i(task) for every task in agent's domain, given a belief `prob`, an
    optional victim claim (cell, confidence) and the agent's shortest-path
    distances `row` (n x n) on its believed map. Returns dict task -> cost."""
    masses = region_masses(building, prob)
    q = agent.p_det * ((2 * agent.radius + 1) ** 2) / 25.0  # scanning efficiency
    costs = {IDLE: 0.0}
    for k, cells in enumerate(building.region_cells):
        d = row[cells].min()
        if not np.isfinite(d) or masses[k] < 1e-6:
            continue
        costs[search(k)] = d / agent.speed - cfg.w_info * q * masses[k]
    if claim is not None and not agent.is_drone:
        cell, conf = claim
        d = row[cell]
        if np.isfinite(d):
            costs[RESCUE] = d / agent.speed - cfg.w_rescue * conf
    return costs


def pair_cost(t1, t2, cfg) -> float:
    """Soft form of constraint (2): at most one agent per non-idle task."""
    if t1 == t2 and t1 != IDLE:
        return cfg.w_dup
    return 0.0


# ---------------------------------------------------------------- centralised
def hungarian(agents, costs_by_agent):
    """Exact solution of (1)-(4) with linear objective: rows = agents,
    columns = distinct non-idle tasks + one private IDLE column per agent."""
    tasks = sorted({t for c in costs_by_agent for t in c if t != IDLE})
    n, m = len(agents), len(tasks)
    M = np.full((n, m + n), BIG)
    for i, (a, c) in enumerate(zip(agents, costs_by_agent)):
        for j, t in enumerate(tasks):
            if t in c and feasible(a, t):
                M[i, j] = c[t]
        M[i, m + i] = c[IDLE]
    r, col = linear_sum_assignment(M)
    out = {}
    for i, j in zip(r, col):
        out[agents[i].id] = tasks[j] if j < m else IDLE
    return out


# ------------------------------------------------------------- decentralised
def local_cost(task, unary, view, me, t, cfg):
    c = unary[task]
    for j, (tj, seen) in view.items():
        if j != me and t - seen <= cfg.view_ttl:
            c += pair_cost(task, tj, cfg)
    return c


def conflicted(task, view, me, t, cfg):
    """True if a fresh neighbour value collides with `task` (DSA-C rule)."""
    return task != IDLE and any(j != me and tj == task and t - seen <= cfg.view_ttl
                                for j, (tj, seen) in view.items())


def best_response(agent, unary, t, cfg):
    best, bc = None, np.inf
    for task in sorted(unary):  # deterministic tie-break
        if not feasible(agent, task):
            continue
        c = local_cost(task, unary, agent.view, agent.id, t, cfg)
        if c < bc:
            best, bc = task, c
    return best, bc
