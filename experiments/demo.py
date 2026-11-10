"""Render snapshots and an animated GIF of one episode.

    python experiments/demo.py --seed 3 --budget 40
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar import Config, Simulation  # noqa: E402
from sar.env import HAZARD, OBSTACLE  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DRONE, ROBOT = "#2a78d6", "#eb6834"


def draw(ax, sim, frame, title):
    b = sim.b
    ax.clear()
    ax.imshow(frame["prob"], cmap="Purples", vmin=0, vmax=max(0.03, frame["prob"].max()),
              origin="upper")
    ov = np.zeros((b.n, b.n, 4))
    ov[b.grid == OBSTACLE] = (0.15, 0.15, 0.15, 1)
    ov[b.grid == HAZARD] = (0.89, 0.29, 0.28, 0.55)
    ax.imshow(ov, origin="upper")
    ax.add_patch(plt.Rectangle((b.exit[1] - .5, b.exit[0] - .5), 1, 1, color="#1baf7a"))
    ax.text(b.exit[1], b.exit[0], "E", ha="center", va="center", fontsize=7, color="white", weight="bold")
    i = frame["t"]
    for k, a in enumerate(sim.agents):
        tr = np.array(a.trajectory[: i + 1])
        ax.plot(tr[:, 1], tr[:, 0], color=DRONE if a.is_drone else ROBOT, lw=0.8, alpha=0.5)
    pos = frame["pos"]
    for p, q in [(p, q) for j, p in enumerate(pos) for q in pos[j + 1:]]:
        if math.dist(p, q) <= sim.cfg.comm_range:
            ax.plot([p[1], q[1]], [p[0], q[0]], ls="--", color="#52514e", lw=0.6)
    if not any(frame["carrying"]):
        ax.plot(b.victim[1], b.victim[0], marker="*", color="#eda100", ms=12, mec="k", mew=0.5)
    for k, (p, kind) in enumerate(zip(pos, frame["kind"])):
        ax.plot(p[1], p[0], marker="^" if kind == "drone" else "s", ms=8,
                color=DRONE if kind == "drone" else ROBOT, mec="white", mew=1)
        ax.text(p[1] + .45, p[0] - .45, ("d" if kind == "drone" else "r") + str(k), fontsize=6)
        if frame["carrying"][k]:
            ax.plot(p[1], p[0], marker="*", color="#eda100", ms=7, mec="k", mew=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}  t={frame['t']}  msgs={frame['comm']}", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--budget", type=int, default=-1)
    ap.add_argument("--method", default="dcop")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    B = None if args.budget < 0 else args.budget
    sim = Simulation(Config(budget=B), args.seed, args.method, record=True)
    res = sim.run()
    print(res)
    name = f"{args.method}_B{'inf' if B is None else B}_s{args.seed}"
    fr = sim.frames
    ev = sim.events
    picks = [0, min(len(fr) - 1, ev.get("detect", len(fr) // 3)),
             min(len(fr) - 1, ev.get("pickup", 2 * len(fr) // 3)), len(fr) - 1]
    labels = ["start", "victim detected", "pickup", "end"]
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, k, lab in zip(axes, picks, labels):
        draw(ax, sim, fr[max(0, k - 1)], lab)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"snapshots_{name}.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    if args.gif:
        fig, ax = plt.subplots(figsize=(4, 4))
        anim = FuncAnimation(fig, lambda k: draw(ax, sim, fr[k], name), frames=len(fr))
        anim.save(os.path.join(FIG, f"episode_{name}.gif"), writer=PillowWriter(fps=6))
        plt.close(fig)


if __name__ == "__main__":
    main()
