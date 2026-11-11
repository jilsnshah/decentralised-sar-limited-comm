"""Discrete-time simulator for both coordination approaches.

method = "central": one coordinator pools every observation for free (every
                    message is still counted), solves (1)-(4) exactly with the
                    Hungarian method and routes agents.
method = "dcop":    every agent keeps its own belief, communicates by
                    range-limited broadcasts charged against its share of the
                    team budget B, and allocates tasks with MGM (or DSA-C).

Unless cfg.known_map is set, the floor plan itself is unknown: agents learn
the terrain inside their sensing radius, plan optimistically through unknown
cells, and share discovered terrain inside their observation messages.
"""
import math
from collections import Counter

import numpy as np
from scipy.ndimage import uniform_filter

from .agent import Agent, Belief
from .allocation import (IDLE, RESCUE, best_response, conflicted, hungarian,
                         local_cost, unary_costs)
from .env import FREE, Building
from .mapknow import MapKnowledge, truly_passable

CARRY = ("carry",)
DELIVER = ("deliver",)


class Simulation:
    def __init__(self, cfg, seed, method="dcop", record=False):
        assert method in ("central", "dcop")
        self.cfg, self.seed, self.method = cfg, seed, method
        self.b = Building(cfg, seed)
        self.rng = np.random.default_rng([seed, 12345])
        self.alloc_rng = np.random.default_rng([seed, 999])
        # victim prior: with a known floor plan the agents can exclude cells a
        # robot cannot reach; otherwise every cell is possible until revealed
        n = self.b.n
        prior = self.b.prior_mask if cfg.known_map else np.ones((n, n), bool)
        self.agents = []
        for i in range(cfg.n_agents):
            kind = "drone" if i < cfg.n_drones else "robot"
            a = Agent(i, kind, self.b.start_cells[i], cfg, prior)
            a.reserve_level = min(cfg.reserve, a.quota)
            a.terrain_out = []
            self.agents.append(a)
        for a in self.agents:
            for r in self.robots:
                a.known_robot_pos[r.id] = r.pos
        if method == "central":
            self.central = Belief(prior)
            self.cmap = MapKnowledge(self.b, cfg.known_map)
            for a in self.agents:
                a.quota = math.inf
                a.reserve_level = 0
                a.map = self.cmap
        else:
            for a in self.agents:
                a.map = MapKnowledge(self.b, cfg.known_map)
        for a in self.agents:  # initial look around the entrance
            ys, xs = self.window(a)
            self.learn_terrain(a, a.map.reveal(ys, xs, self.b.grid))
        self.t = 0
        self.comm = 0
        self.msgs = Counter()
        self.move = 0
        self.success = False
        self.rescued = False
        self.events = {}
        self.dup_steps = 0
        self.dup_pairs = 0
        self.false_trips = 0
        self.wrong_claims = 0
        self.scanned_by = np.zeros((self.b.n, self.b.n), dtype=np.int64)
        self.record = record
        self.frames = []

    # ------------------------------------------------------------- helpers
    @property
    def robots(self):
        return [a for a in self.agents if not a.is_drone]

    def prob_of(self, a):
        if self.method == "central":
            return self._central_prob
        return a._prob

    def window(self, a):
        n, (y, x), r = self.b.n, a.pos, a.radius
        ys, xs = np.mgrid[max(0, y - r): min(n, y + r + 1), max(0, x - r): min(n, x + r + 1)]
        return ys.ravel(), xs.ravel()

    def learn_terrain(self, a, new):
        """Cells newly revealed to agent a: the victim cannot be in an
        obstacle or hazard; in the decentralised case the discovery is queued
        for sharing."""
        if not new:
            return
        blocked = [(y, x) for y, x, v in new if v != FREE]
        if self.method == "central":
            if blocked:
                self.central.clear(blocked)
        else:
            if blocked:
                a.belief.clear(blocked)
            a.terrain_out.extend(new)

    def dist(self, a):
        return a.map.plan(a.pos, a.kind)[0]

    def in_range(self, a, b):
        return math.dist(a.pos, b.pos) <= self.cfg.comm_range

    def note(self, key):
        self.events.setdefault(key, self.t)

    # ------------------------------------------------------- communication
    def broadcast(self, sender, mtype, payload, urgent=False):
        """Range-limited broadcast; costs one unit of the sender's quota."""
        if not sender.can_send(urgent):
            return None
        sender.charge()
        self.comm += 1
        self.msgs[mtype] += 1
        rec = [a for a in self.agents if a is not sender and self.in_range(a, sender)]
        for r in rec:
            self.receive(sender, r, mtype, payload)
        return rec

    def receive(self, s, r, mtype, p):
        t = self.t
        if "task" in p:
            r.view[s.id] = (p["task"], t)
        if not s.is_drone:
            r.known_robot_pos[s.id] = s.pos
        if mtype == "GAIN":
            r.gains_heard[s.id] = p["gain"]
            return
        if mtype != "OBS":
            return
        for ys, xs, z, pd, pf in p.get("obs", ()):
            r.belief.update(ys, xs, z, pd, pf)
        if p.get("terrain"):
            r.map.merge(p["terrain"])
            blocked = [(y, x) for y, x, v in p["terrain"] if v != FREE]
            if blocked:
                r.belief.clear(blocked)
        if p.get("cleared"):
            r.belief.clear(p["cleared"])
            cl = set(p["cleared"])
            for k in [k for k, v in r.claims.items() if v[0] in cl]:
                del r.claims[k]
        if "claim" in p:
            if p["claim"] is None:
                r.claims.pop(s.id, None)
            else:
                old = r.claims.get(s.id)
                r.claims[s.id] = (p["claim"][0], p["claim"][1], t)
                if old is None or old[0] != p["claim"][0]:
                    r.relay_pending = p["claim"]
        if p.get("rescued"):
            r.rescued = True

    def flush_obs(self, a, urgent=False, extra=None):
        payload = {"obs": a.outbox, "cleared": a.cleared, "task": a.task,
                   "claim": a.own_claim, "terrain": a.terrain_out}
        if extra:
            payload.update(extra)
        rec = self.broadcast(a, "OBS", payload, urgent)
        if rec is not None:
            a.outbox, a.cleared, a.terrain_out = [], [], []
            a.last_claim_sent = a.own_claim
        return rec

    # ------------------------------------------------------------- claims
    def active_claim(self, a):
        """Conflict-resolution rule: among the agent's own MAP estimate (if its
        posterior exceeds tau) and the claims it has received, keep the one with
        the highest confidence; ties go to the most recent report. Claims on
        cells the agent knows to be empty are discarded."""
        if self.method == "central":
            cell, conf = self._central_map
            return (cell, conf) if conf >= self.cfg.tau else None
        cands = []
        if a.own_claim is not None:
            cands.append((a.own_claim[1], self.t, a.own_claim[0]))
        for cell, conf, t in a.claims.values():
            if np.isfinite(a.belief.logp[cell]):
                cands.append((conf, t, cell))
        if not cands:
            return None
        conf, _, cell = max(cands)
        return (cell, conf)

    # ---------------------------------------------------------- targeting
    def choose_target(self, a, prob):
        task = a.task
        if task == CARRY:
            return self.b.exit
        if task == RESCUE and a.claim is not None:
            return a.claim[0]
        if task == DELIVER:
            known = list(a.known_robot_pos.values())
            D = self.dist(a)
            return min(known, key=lambda p: D[p])
        if task is None or task[0] != "search":
            return a.pos
        cells = self.b.region_cells[task[1]]
        w = 2 * a.radius + 1
        S = uniform_filter(prob, size=w, mode="constant") * w * w
        D = self.dist(a)
        ok = cells & np.isfinite(D)
        if not ok.any():
            return a.pos
        score = np.where(ok, S / (1.0 + 0.1 * D), -1.0)
        k = int(np.argmax(score))
        return divmod(k, self.b.n)

    # ---------------------------------------------------------- allocation
    def fixed_task(self, a):
        if a.carrying:
            return CARRY
        if self.rescued:
            return IDLE
        if self.method == "dcop" and getattr(a, "rescued", False):
            return IDLE
        if (self.method == "dcop" and a.is_drone and a.claim is not None
                and a.own_claim is not None and a.own_claim[0] == a.claim[0]
                and a.claim[0] not in a.delivered_cells and a.quota - a.sent > 0
                and a.known_robot_pos):
            return DELIVER
        return None

    def allocate_central(self):
        free, costs = [], []
        for a in self.agents:
            f = self.fixed_task(a)
            if f is not None:
                a.task = f
                continue
            free.append(a)
            costs.append(unary_costs(a, self.b, self.cfg, self._central_prob, a.claim, self.dist(a)))
        if free:
            assign = hungarian(free, costs)
            for a in free:
                if a.task != assign[a.id]:
                    self.comm += 1          # coordinator -> agent assignment
                    self.msgs["ASSIGN"] += 1
                a.task = assign[a.id]

    def allocate_dcop(self, full):
        cfg, t = self.cfg, self.t
        active = []
        for a in self.agents:
            f = self.fixed_task(a)
            if f is not None:
                a.task = f
                continue
            a.unary = unary_costs(a, self.b, cfg, a._prob, a.claim, self.dist(a))
            if a.task not in a.unary:
                a.task, _ = best_response(a, a.unary, t, cfg)
            active.append(a)
        if not full:
            # between MGM rounds only agents hit by a new claim re-decide locally
            for a in active:
                if a.claim_event:
                    a.task, _ = best_response(a, a.unary, t, cfg)
            return
        if cfg.solver == "none" or cfg.rounds == 0:
            # ablation: no coordination messages, independent best responses
            # using whatever task information was piggybacked on OBS messages
            for a in active:
                a.task, _ = best_response(a, a.unary, t, cfg)
            return
        for _ in range(cfg.rounds):
            # VALUE phase: announce current assignment (if changed or stale)
            for a in active:
                if a.task != a.announced or t - a.announced_t >= cfg.view_ttl:
                    if self.broadcast(a, "VALUE", {"task": a.task}) is not None:
                        a.announced, a.announced_t = a.task, t
            if cfg.solver == "dsa":
                # DSA-C: move to the best response with probability p if it
                # strictly improves, or if it keeps the cost while the current
                # value is in conflict. No gain messages are exchanged.
                changed = False
                for a in active:
                    best, bc = best_response(a, a.unary, t, cfg)
                    cur = local_cost(a.task, a.unary, a.view, a.id, t, cfg)
                    gain = cur - bc
                    move = gain > 1e-9 or (abs(gain) <= 1e-9 and best != a.task
                                           and conflicted(a.task, a.view, a.id, t, cfg))
                    if move and self.alloc_rng.random() < cfg.dsa_p:
                        a.task = best
                        changed = True
                continue
            # GAIN phase
            for a in active:
                a.gains_heard = {}
            prop = {}
            for a in active:
                best, bc = best_response(a, a.unary, t, cfg)
                cur = local_cost(a.task, a.unary, a.view, a.id, t, cfg)
                gain = cur - bc
                prop[a.id] = (best, gain)
                if gain > 1e-9:
                    self.broadcast(a, "GAIN", {"gain": gain})
            # MOVE phase: only the locally-maximal-gain agent changes value
            changed = False
            for a in active:
                best, gain = prop[a.id]
                if gain <= 1e-9:
                    continue
                rivals = [(g, -j) for j, g in a.gains_heard.items()]
                if all((gain, -a.id) > r for r in rivals):
                    a.task = best
                    changed = True
            if not changed:
                break

    # ------------------------------------------------------------- the loop
    def step(self):
        cfg, b = self.cfg, self.b
        self.t += 1
        t = self.t
        # 1. beliefs and claims
        if self.method == "central":
            self._central_prob = self.central.prob()
            k = int(np.argmax(self._central_prob))
            self._central_map = (divmod(k, b.n), float(self._central_prob.flat[k]))
        for a in self.agents:
            if self.method == "dcop":
                a._prob = a.belief.prob()
                k = int(np.argmax(a._prob))
                cell, conf = divmod(k, b.n), float(a._prob.flat[k])
                a.own_claim = (cell, conf) if conf >= cfg.tau else None
            new = self.active_claim(a)
            a.claim_event = (new is not None and (a.claim is None or new[0] != a.claim[0]))
            a.claim = new
            if new is not None and new[0] == b.victim:
                self.note("detect")
                if not a.is_drone:
                    self.note("handoff")
        # 2. task allocation
        replan = (t - 1) % cfg.replan_period == 0
        if self.method == "central":
            if replan or any(a.claim_event for a in self.agents):
                self.allocate_central()
            for a in self.agents:
                f = self.fixed_task(a)
                if f is not None:
                    a.task = f
        else:
            self.allocate_dcop(full=replan)
        # constraint (2) audit
        tasks = [a.task for a in self.agents if a.task not in (IDLE, CARRY, DELIVER, None)]
        dup = len(tasks) - len(set(tasks))
        if dup:
            self.dup_steps += 1
            self.dup_pairs += dup
        # 3. motion
        for a in self.agents:
            prob = self.prob_of(a)
            a.target = self.choose_target(a, prob)
            path = a.map.path(a.pos, a.target, a.kind)
            for nxt in path[: a.speed]:
                if not truly_passable(b, nxt, a.kind):
                    # bumped into terrain that was not known yet
                    new = a.map.reveal(np.array([nxt[0]]), np.array([nxt[1]]), b.grid)
                    self.learn_terrain(a, new)
                    break
                a.pos = nxt
                self.move += 1
            a.trajectory.append(a.pos)
        # 4. sensing, physical contact, rescue
        for a in self.agents:
            ys, xs, z = a.scan(b, self.rng)
            self.learn_terrain(a, a.map.reveal(ys, xs, b.grid))
            self.scanned_by[ys, xs] |= (1 << a.id)
            if self.method == "central":
                self.central.update(ys, xs, z, a.p_det, a.p_fa)
                self.comm += 1                   # agent -> coordinator upload
                self.msgs["OBS"] += 1
            else:
                a.outbox.append((ys, xs, z, a.p_det, a.p_fa))
            if a.is_drone:
                continue
            if a.carrying:
                if a.pos == b.exit:
                    self.success = True
                continue
            if a.pos == b.victim and not self.rescued:
                a.carrying = True
                a.task = CARRY
                self.rescued = True
                self.note("pickup")
                if self.method == "dcop":
                    a.rescued = True
                    self.flush_obs(a, urgent=True, extra={"rescued": True})
            else:
                was_claimed = a.claim is not None and a.claim[0] == a.pos
                if was_claimed:
                    self.false_trips += 1
                if self.method == "central":
                    self.central.clear([a.pos])
                else:
                    a.belief.clear([a.pos])
                    a.cleared.append(a.pos)
                    for k in [k for k, v in a.claims.items() if v[0] == a.pos]:
                        del a.claims[k]
                    if was_claimed:
                        self.flush_obs(a, urgent=True)
        # 5. communication (decentralised only)
        if self.method == "dcop":
            self.communicate()
        if self.record:
            self.snapshot()

    def communicate(self):
        cfg, t, b = self.cfg, self.t, self.b
        for a in self.agents:
            a._prob = a.belief.prob()
            k = int(np.argmax(a._prob))
            cell, conf = divmod(k, b.n), float(a._prob.flat[k])
            a.own_claim = (cell, conf) if conf >= cfg.tau else None
            sent = False
            # (i) urgent: a new or retracted own claim
            if a.own_claim is not None and (a.last_claim_sent is None
                                            or a.last_claim_sent[0] != a.own_claim[0]):
                if a.own_claim[0] != b.victim:
                    self.wrong_claims += 1
                rec = self.flush_obs(a, urgent=True)
                sent = rec is not None
                if sent and any(not r.is_drone for r in rec):
                    a.delivered_cells.add(a.own_claim[0])
            elif a.own_claim is None and a.last_claim_sent is not None:
                sent = self.flush_obs(a, urgent=False) is not None
            # (ii) drone delivering a claim to a robot in range
            if not sent and a.task == DELIVER and a.claim is not None:
                if any(not r.is_drone and self.in_range(a, r) for r in self.agents):
                    payload = {"task": a.task, "claim": a.claim, "obs": a.outbox,
                               "cleared": a.cleared, "terrain": a.terrain_out}
                    rec = self.broadcast(a, "OBS", payload, urgent=True)
                    if rec is not None:
                        a.outbox, a.cleared, a.terrain_out = [], [], []
                        a.delivered_cells.add(a.claim[0])
                        sent = True
            # (iii) relay a newly heard claim once
            if not sent and a.relay_pending is not None:
                cl = a.relay_pending
                a.relay_pending = None
                if cl[0] not in a.relayed:
                    a.relayed.add(cl[0])
                    rec = self.broadcast(a, "OBS", {"task": a.task, "claim": cl}, urgent=True)
                    if rec is not None:
                        sent = True
                        if any(not r.is_drone for r in rec):
                            a.delivered_cells.add(cl[0])
            # (iv) periodic sharing of observations
            if not sent and t % cfg.share_period == 0 and a.outbox:
                self.flush_obs(a, urgent=False)

    def snapshot(self):
        self.frames.append({
            "t": self.t,
            "pos": [a.pos for a in self.agents],
            "task": [a.task for a in self.agents],
            "kind": [a.kind for a in self.agents],
            "prob": (self._central_prob if self.method == "central" else self.agents[0]._prob).copy(),
            "carrying": [a.carrying for a in self.agents],
            "comm": self.comm,
            "known": self.agents[0].map.grid.copy(),
        })

    def run(self):
        while self.t < self.cfg.max_steps and not self.success:
            self.step()
        return self.result()

    def result(self):
        cfg = self.cfg
        T = self.t if self.success else cfg.max_steps
        scanned = self.scanned_by > 0
        multi = np.array([bin(int(v)).count("1") >= 2 for v in self.scanned_by.ravel()])
        redundancy = multi.sum() / max(1, scanned.sum())
        known = np.zeros((self.b.n, self.b.n), bool)
        for a in self.agents:
            known |= a.map.grid >= 0
        J = (cfg.lam_t * T + cfg.lam_c * self.comm + cfg.lam_m * self.move
             + cfg.lam_f * (1 - int(self.success)))
        return {
            "method": self.method, "seed": self.seed,
            "budget": -1 if cfg.budget is None else cfg.budget,
            "success": int(self.success), "T": T, "comm": self.comm, "move": self.move,
            "J": J,
            "t_detect": self.events.get("detect", np.nan),
            "t_handoff": self.events.get("handoff", np.nan),
            "t_pickup": self.events.get("pickup", np.nan),
            "dup_steps": self.dup_steps, "dup_pairs": self.dup_pairs,
            "false_trips": self.false_trips, "wrong_claims": self.wrong_claims,
            "redundancy": redundancy,
            "explored": float(known.mean()),
            "msg_obs": self.msgs["OBS"], "msg_value": self.msgs["VALUE"],
            "msg_gain": self.msgs["GAIN"], "msg_assign": self.msgs["ASSIGN"],
            "max_agent_sent": max(a.sent for a in self.agents),
        }


def run_episode(cfg, seed, method):
    return Simulation(cfg, seed, method).run()
