"""Checks that the implementation matches the formal model of Section 3.

    python -m pytest tests -q        (or: python tests/test_model.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar import Config, Simulation, run_episode  # noqa: E402
from sar.agent import Belief  # noqa: E402
from sar.allocation import IDLE, RESCUE, feasible, hungarian, unary_costs  # noqa: E402
from sar.env import Building, HAZARD, OBSTACLE  # noqa: E402


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
    costs = [unary_costs(a, sim.b, cfg, prob, claim) for a in sim.agents]
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
