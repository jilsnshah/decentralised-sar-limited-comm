"""Experiment configuration and sensor presets."""
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

# Sensor reliability presets: (p_det, p_fa) for drones and ground robots.
# Drones always dominate robots (p_det higher, radius larger), as required by
# the observation model (Eq. 5 of the report).
RELIABILITY = {
    "high":   {"drone": (0.95, 0.01), "robot": (0.70, 0.02)},
    "medium": {"drone": (0.80, 0.03), "robot": (0.55, 0.04)},
    "low":    {"drone": (0.60, 0.06), "robot": (0.45, 0.07)},
}

# Building-complexity presets: (grid side, obstacle density, hazard density, n internal walls)
COMPLEXITY = {
    "simple":  (15, 0.05, 0.03, 2),
    "medium":  (20, 0.08, 0.06, 3),
    "complex": (28, 0.10, 0.08, 5),
}

INF = None  # budget value meaning "unrestricted"


@dataclass(frozen=True)
class Config:
    # environment
    size: int = 20
    obstacle_density: float = 0.08
    hazard_density: float = 0.06
    n_walls: int = 3
    # team
    n_drones: int = 3
    n_robots: int = 2
    drone_speed: int = 2
    robot_speed: int = 1
    drone_radius: int = 2
    robot_radius: int = 1
    reliability: str = "medium"
    # communication
    budget: Optional[int] = INF       # team budget B (None = unrestricted)
    comm_range: float = 9.0           # Euclidean broadcast range (cells)
    share_period: int = 5             # periodic observation-sharing period (steps)
    reserve: int = 2                  # per-agent quota reserved for victim claims
    # allocation
    regions: Tuple[int, int] = (4, 4)  # search-region partition (rows, cols)
    replan_period: int = 5
    mgm_rounds: int = 3
    view_ttl: int = 15                # staleness limit for neighbours' task info
    w_info: float = 400.0             # weight on searchable belief mass
    w_rescue: float = 1000.0          # weight on rescuing a claimed victim
    w_dup: float = 1000.0             # soft form of constraint (2) in the DCOP
    tau: float = 0.7                  # posterior threshold for a victim claim
    # episode
    max_steps: int = 150
    # objective weights (Eq. 7)
    lam_t: float = 1.0
    lam_c: float = 0.1
    lam_m: float = 0.05
    lam_f: float = 300.0

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    @property
    def n_agents(self) -> int:
        return self.n_drones + self.n_robots

    def sensor(self, kind: str):
        return RELIABILITY[self.reliability][kind]

    def complexity(self, level: str) -> "Config":
        s, od, hd, nw = COMPLEXITY[level]
        return self.with_(size=s, obstacle_density=od, hazard_density=hd, n_walls=nw)
