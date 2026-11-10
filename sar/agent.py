"""Heterogeneous agents with a Bayesian belief over the victim location."""
import math
import numpy as np


class Belief:
    """Posterior Pr(v* = v | evidence) over grid cells, stored in log space.

    A reading z_v from a sensor with parameters (p_det, p_fa) multiplies the
    posterior at v by the likelihood ratio L(z) = Pr(z | v=v*) / Pr(z | v!=v*)
    (all other cells share the denominator, which cancels on normalisation).
    """

    def __init__(self, prior_mask: np.ndarray):
        self.logp = np.where(prior_mask, 0.0, -np.inf)

    def copy(self):
        b = Belief.__new__(Belief)
        b.logp = self.logp.copy()
        return b

    def update(self, ys, xs, z, p_det, p_fa):
        lr_pos = math.log(p_det / p_fa)
        lr_neg = math.log((1 - p_det) / (1 - p_fa))
        self.logp[ys, xs] += np.where(z, lr_pos, lr_neg)

    def clear(self, cells):
        """Certain absence (a ground robot physically checked the cell)."""
        for (y, x) in cells:
            self.logp[y, x] = -np.inf

    def prob(self) -> np.ndarray:
        m = self.logp.max()
        if not np.isfinite(m):
            return np.zeros_like(self.logp)
        p = np.exp(self.logp - m)
        return p / p.sum()

    def map_estimate(self):
        p = self.prob()
        k = int(np.argmax(p))
        return divmod(k, p.shape[1]), float(p.flat[k])


class Agent:
    def __init__(self, aid, kind, pos, cfg, prior_mask):
        self.id = aid
        self.kind = kind  # "drone" | "robot"
        self.pos = pos
        self.speed = cfg.drone_speed if kind == "drone" else cfg.robot_speed
        self.radius = cfg.drone_radius if kind == "drone" else cfg.robot_radius
        self.p_det, self.p_fa = cfg.sensor(kind)
        self.belief = Belief(prior_mask)
        self.task = None
        self.target = pos
        self.carrying = False
        # communication state
        self.quota = math.inf if cfg.budget is None else cfg.budget // cfg.n_agents
        self.sent = 0
        self.outbox = []          # unshared observations: (sender_kind_params, ys, xs, z)
        self.cleared = []         # unshared certain-absence cells
        self.view = {}            # agent id -> (task, time seen)
        self.gains_heard = {}
        self.claims = {}          # cell -> (conf, time)  (victim reports received)
        self.last_claim_sent = None
        self.delivered = False
        self.known_robot_pos = {}
        self.trajectory = [pos]
        self.claim = None
        self.own_claim = None
        self.claim_event = False
        self.relay_pending = None
        self.relayed = set()
        self.delivered_cells = set()
        self.announced, self.announced_t = None, -10 ** 9
        self.rescued = False
        self.unary = {}
        self.reserve_level = 0
        self._prob = None

    @property
    def is_drone(self):
        return self.kind == "drone"

    def scan(self, building, rng):
        """Noisy local observation (Eq. 5) of every cell within the sensing radius."""
        n = building.n
        y, x = self.pos
        r = self.radius
        ys, xs = np.mgrid[max(0, y - r): min(n, y + r + 1), max(0, x - r): min(n, x + r + 1)]
        ys, xs = ys.ravel(), xs.ravel()
        is_v = (ys == building.victim[0]) & (xs == building.victim[1])
        p1 = np.where(is_v, self.p_det, self.p_fa)
        z = rng.random(len(ys)) < p1
        self.belief.update(ys, xs, z, self.p_det, self.p_fa)
        return ys, xs, z

    def can_send(self, urgent=False) -> bool:
        left = self.quota - self.sent
        return left > 0 if urgent else left > self.reserve_level

    def charge(self, c=1):
        self.sent += c
