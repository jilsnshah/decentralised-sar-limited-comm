"""Run every experiment of the report and write one CSV per experiment to results/.

    python experiments/run_experiments.py            # all experiments, 200 seeds
    python experiments/run_experiments.py --seeds 20 --only main
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar import Config, run_episode  # noqa: E402

BUDGETS = [0, 10, 20, 40, 80, 160, 320, None]
COARSE = [0, 20, 80, None]          # none / low / moderate / unrestricted
BASE = Config()


def jobs_main(seeds):
    for s in seeds:
        yield ("main", {}, "central", BASE, s)
        for B in BUDGETS:
            yield ("main", {}, "dcop", BASE.with_(budget=B), s)
            yield ("main", {}, "dcop-nomgm", BASE.with_(budget=B, mgm_rounds=0), s)


def jobs_reliability(seeds):
    for rel in ("high", "medium", "low"):
        c = BASE.with_(reliability=rel)
        for s in seeds:
            yield ("reliability", {"reliability": rel}, "central", c, s)
            for B in BUDGETS:
                yield ("reliability", {"reliability": rel}, "dcop", c.with_(budget=B), s)


def jobs_team(seeds):
    for nd, nr in ((2, 1), (3, 2), (5, 3)):
        c = BASE.with_(n_drones=nd, n_robots=nr)
        tag = {"team": f"{nd}D+{nr}R", "n_agents": nd + nr}
        for s in seeds:
            yield ("team", tag, "central", c, s)
            for B in COARSE:
                yield ("team", tag, "dcop", c.with_(budget=B), s)


def jobs_complexity(seeds):
    for lvl in ("simple", "medium", "complex"):
        c = BASE.complexity(lvl)
        for s in seeds:
            yield ("complexity", {"complexity": lvl}, "central", c, s)
            for B in COARSE:
                yield ("complexity", {"complexity": lvl}, "dcop", c.with_(budget=B), s)


def jobs_range(seeds):
    for R in (4.0, 6.0, 9.0, 14.0, 100.0):
        for s in seeds:
            for B in (40, None):
                yield ("range", {"comm_range": R}, "dcop", BASE.with_(budget=B, comm_range=R), s)


EXPERIMENTS = {
    "main": jobs_main,
    "reliability": jobs_reliability,
    "team": jobs_team,
    "complexity": jobs_complexity,
    "range": jobs_range,
}


def work(job):
    exp, tag, label, cfg, seed = job
    method = "central" if label == "central" else "dcop"
    r = run_episode(cfg, seed, method)
    r.update(tag)
    r["method"] = label
    return exp, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = range(args.seeds)
    for name in args.only:
        jobs = list(EXPERIMENTS[name](seeds))
        t0 = time.time()
        with Pool(args.procs) as pool:
            rows = [r for _, r in pool.imap_unordered(work, jobs, chunksize=8)]
        df = pd.DataFrame(rows).sort_values(["method", "budget", "seed"])
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name}: {len(jobs)} episodes in {time.time() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
