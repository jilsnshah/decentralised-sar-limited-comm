"""Checks that the implementation matches the formal model of Section 3.

    python -m pytest tests -q        (or: python tests/test_model.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar import Config, Simulation, run_episode  # noqa: E402
from sar.agent import Belief  # noqa: E402
import itertools  # noqa: E402

from sar.allocation import IDLE, RESCUE, feasible, hungarian, search, unary_costs  # noqa: E402
from sar.env import Building, FREE, HAZARD, OBSTACLE  # noqa: E402
from sar.mapknow import UNKNOWN, MapKnowledge  # noqa: E402


def test_building_valid():
    cfg = Config()
    for s in range(20):
        b = Building(cfg, s)
        assert b.grid[b.victim] == 0 and b.robot_reach[b.victim]
        assert b.grid[b.exit] != OBSTACLE
        # robots never route through hazards, drones never through walls
        assert not np.isfinite(b.dist["robot"][:, (b.grid == HAZARD).ravel()]).any()
        assert not np.isfinite(b.dist["drone"][:, (b.grid == OBSTACLE).ravel()]).any()


def test_bayes_update_matches_closed_form():
    mask = np.ones((3, 3), bool)
    bel = Belief(mask)
    bel.update(np.array([1]), np.array([1]), np.array([True]), 0.8, 0.05)
    p = bel.prob()
    # posterior of the positive cell: 0.8 / (0.8 + 8*0.05)
    assert abs(p[1, 1] - 0.8 / (0.8 + 8 * 0.05)) < 1e-9


def test_capability_constraint_4():
    cfg = Config()
    sim = Simulation(cfg, 0, "central")
    drone, robot = sim.agents[0], sim.agents[-1]
    assert drone.is_drone and not robot.is_drone
    assert not feasible(drone, RESCUE) and feasible(robot, RESCUE)
    prob = sim.b.prior_mask / sim.b.prior_mask.sum()
    claim = (sim.b.victim, 0.9)
    costs = [unary_costs(a, sim.b, cfg, prob, claim, sim.dist(a)) for a in sim.agents]
    assign = hungarian(sim.agents, costs)
    # (2) every non-idle task held by at most one agent; (4) rescue only by robots
    held = [t for t in assign.values() if t != IDLE]
    assert len(held) == len(set(held))
    assert all(not sim.agents[i].is_drone for i, t in assign.items() if t == RESCUE)
    assert RESCUE in assign.values()


def test_budget_constraint_6():
    for B in (0, 10, 40):
        for s in range(15):
            r = run_episode(Config(budget=B), s, "dcop")
            assert r["comm"] <= B
            assert r["max_agent_sent"] <= B // Config().n_agents


def test_centralised_never_violates_2():
    for s in range(15):
        assert run_episode(Config(), s, "central")["dup_steps"] == 0


def test_zero_budget_means_no_messages():
    r = run_episode(Config(budget=0), 1, "dcop")
    assert r["comm"] == 0 and r["msg_obs"] == 0


def test_determinism():
    a = run_episode(Config(budget=40), 7, "dcop")
    b = run_episode(Config(budget=40), 7, "dcop")
    assert a == b or all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)


def test_hungarian_is_optimal_bruteforce():
    """Exact optimality of the centralised solver against enumeration of every
    assignment satisfying (2)-(4)."""
    sim = Simulation(Config(), 0, "central")
    agents = sim.agents[:4]
    rng = np.random.default_rng(0)
    tasks = [search(k) for k in range(4)] + [RESCUE]
    for _ in range(50):
        costs = []
        for a in agents:
            c = {IDLE: 0.0}
            for t in tasks:
                if rng.random() < 0.85:
                    c[t] = float(rng.normal(-5, 10))
            costs.append(c)
        assign = hungarian(agents, costs)
        got = sum(costs[i][assign[a.id]] for i, a in enumerate(agents))
        best = np.inf
        for combo in itertools.product(*[list(c) for c in costs]):
            held = [t for t in combo if t != IDLE]
            if len(held) != len(set(held)):
                continue
            if any(not feasible(a, t) for a, t in zip(agents, combo)):
                continue
            best = min(best, sum(c[t] for c, t in zip(costs, combo)))
        assert abs(got - best) < 1e-9


def test_unknown_map_is_discovered_and_shared():
    cfg = Config(budget=None)
    sim = Simulation(cfg, 2, "dcop")
    a = sim.agents[0]
    assert (a.map.grid == UNKNOWN).mean() > 0.8          # starts mostly unknown
    for _ in range(30):
        sim.step()
    known = a.map.grid != UNKNOWN
    # everything an agent believes about terrain is true
    for ag in sim.agents:
        k = ag.map.grid != UNKNOWN
        assert (ag.map.grid[k] == sim.b.grid[k]).all()
    assert known.mean() > 0.3
    # robots learnt terrain they never saw themselves (via OBS messages)
    r = sim.agents[-1]
    seen = np.zeros_like(known)
    for (y, x) in r.trajectory:
        seen[max(0, y - 1): y + 2, max(0, x - 1): x + 2] = True
    assert ((r.map.grid != UNKNOWN) & ~seen).any()


def test_agents_never_enter_forbidden_cells():
    for method in ("central", "dcop"):
        sim = Simulation(Config(), 4, method)
        sim.run()
        for a in sim.agents:
            for p in a.trajectory:
                assert sim.b.grid[p] != OBSTACLE
                if not a.is_drone:
                    assert sim.b.grid[p] == FREE


def test_dsa_respects_budget_and_runs():
    for B in (0, 20, None):
        r = run_episode(Config(budget=B, solver="dsa"), 3, "dcop")
        assert r["msg_gain"] == 0
        if B is not None:
            assert r["comm"] <= B


def test_known_map_mode():
    sim = Simulation(Config(known_map=True), 0, "dcop")
    assert (sim.agents[0].map.grid == sim.b.grid).all()
    assert MapKnowledge(sim.b, known=False).grid[sim.b.exit] == FREE


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
