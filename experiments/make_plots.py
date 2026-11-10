"""Turn results/*.csv into the figures and LaTeX tables used in the report.

    python experiments/make_plots.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 4.5, "savefig.dpi": 200,
    "savefig.bbox": "tight", "font.family": "serif",
})

BUDGET_ORDER = [0, 10, 20, 40, 80, 160, 320, -1]
BLAB = {b: ("∞" if b == -1 else str(b)) for b in BUDGET_ORDER}


def load(name):
    return pd.read_csv(os.path.join(RES, f"{name}.csv"))


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def metric(df, col):
    if col == "T_succ":
        return df.loc[df.success == 1, "T"]
    return df[col]


def budget_curve(ax, df, col, color, label, marker="o"):
    xs, ms, cs = [], [], []
    for i, b in enumerate(BUDGET_ORDER):
        sub = df[df.budget == b]
        if len(sub) == 0:
            continue
        m, c = mean_ci(metric(sub, col))
        xs.append(i); ms.append(m); cs.append(c)
    xs, ms, cs = map(np.array, (xs, ms, cs))
    ax.fill_between(xs, ms - cs, ms + cs, color=color, alpha=0.15, lw=0)
    ax.plot(xs, ms, marker=marker, color=color, label=label)


def ref_line(ax, df, col, color=INK, label="centralised", ls="--"):
    m, _ = mean_ci(metric(df, col))
    ax.axhline(m, color=color, ls=ls, lw=1.3, label=label)


def budget_axis(ax):
    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BLAB[b] for b in BUDGET_ORDER])
    ax.set_xlabel("budget $B$ (messages)")


# ------------------------------------------------------------------ figures
def fig_main():
    df = load("main")
    cen = df[df.method == "central"]
    panels = [("success", "rescue success rate"), ("T_succ", "completion time $T_{rescue}$ (succ.)"),
              ("move", "movement cost $C_{move}$"), ("t_handoff", "handoff time (robot learns $v^*$)"),
              ("comm", "messages used $C_{comm}$"), ("dup_steps", "steps violating constraint (2)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    for ax, (col, title) in zip(axes.ravel(), panels):
        budget_curve(ax, df[df.method == "dcop"], col, BLUE, "DCOP (MGM)")
        budget_curve(ax, df[df.method == "dcop-nomgm"], col, ORANGE, "no-MGM ablation", "s")
        ref_line(ax, cen, col)
        ax.set_title(title)
        budget_axis(ax)
    axes[0, 0].set_ylim(0.7, 1.02)
    for ax in axes[0]:
        ax.set_xlabel("")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "budget_main.png"))
    plt.close(fig)


def fig_reliability():
    df = load("reliability")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    cols = {"high": BLUE, "medium": ORANGE, "low": AQUA}
    for rel, c in cols.items():
        d = df[df.reliability == rel]
        for ax, col in zip(axes, ("success", "T", "false_trips")):
            budget_curve(ax, d[d.method == "dcop"], col, c, f"{rel} reliability")
            ref_line(ax, d[d.method == "central"], col, color=c, label=None, ls=":")
    for ax, t in zip(axes, ("success rate", "$T_{rescue}$ (all runs, cap 150)", "robot trips to false claims")):
        ax.set_title(t)
        budget_axis(ax)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reliability.png"))
    plt.close(fig)


def grouped_bars(ax, df, key, levels, col):
    budgets = [0, 20, 80, -1]
    colors = [VIOLET, ORANGE, AQUA, BLUE]
    names = ["B=0", "B=20", "B=80", "B=∞"]
    w = 0.16
    for k, lev in enumerate(levels):
        d = df[df[key] == lev]
        for j, (b, c) in enumerate(zip(budgets, colors)):
            m, ci = mean_ci(metric(d[(d.method == "dcop") & (d.budget == b)], col))
            ax.bar(k + (j - 1.5) * w - 0.08, m, w * 0.88, yerr=ci, color=c,
                   label=names[j] if k == 0 else None, error_kw={"lw": 0.8, "ecolor": INK2})
        m, ci = mean_ci(metric(d[d.method == "central"], col))
        ax.bar(k + 2.5 * w - 0.08, m, w * 0.88, yerr=ci, color="#9a9890",
               label="centralised" if k == 0 else None, error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_xticks(range(len(levels)))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)


def fig_team_complexity():
    tm, cx = load("team"), load("complexity")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3))
    tl = ["2D+1R", "3D+2R", "5D+3R"]
    cl = ["simple", "medium", "complex"]
    grouped_bars(axes[0], tm, "team", tl, "success"); axes[0].set_xticklabels(tl)
    grouped_bars(axes[1], tm, "team", tl, "T"); axes[1].set_xticklabels(tl)
    grouped_bars(axes[2], cx, "complexity", cl, "success"); axes[2].set_xticklabels(cl)
    grouped_bars(axes[3], cx, "complexity", cl, "T"); axes[3].set_xticklabels(cl)
    axes[0].set_title("team size: success"); axes[1].set_title("team size: $T_{rescue}$")
    axes[2].set_title("complexity: success"); axes[3].set_title("complexity: $T_{rescue}$")
    axes[0].set_ylim(0, 1.05); axes[2].set_ylim(0, 1.05)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "team_complexity.png"))
    plt.close(fig)


def fig_range():
    df = load("range")
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
    for b, c, lab in ((-1, BLUE, "B=∞"), (40, ORANGE, "B=40")):
        d = df[df.budget == b]
        rs = sorted(d.comm_range.unique())
        for ax, col in zip(axes, ("T", "dup_steps")):
            mc = [mean_ci(d[d.comm_range == r][col]) for r in rs]
            m = np.array([x[0] for x in mc]); ci = np.array([x[1] for x in mc])
            xs = np.arange(len(rs))
            ax.fill_between(xs, m - ci, m + ci, color=c, alpha=0.15, lw=0)
            ax.plot(xs, m, marker="o", color=c, label=lab)
            ax.set_xticks(xs)
            ax.set_xticklabels(["∞" if r >= 100 else f"{r:g}" for r in rs])
            ax.set_xlabel("communication range (cells)")
    axes[0].set_title("$T_{rescue}$"); axes[1].set_title("steps violating (2)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "range.png"))
    plt.close(fig)


def fig_tradeoff():
    df = load("main")
    d = df[df.method == "dcop"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xs, ys = [], []
    for b in BUDGET_ORDER:
        s = d[d.budget == b]
        x, y = s.comm.mean(), s.J.mean() - 0.1 * s.comm.mean()
        xs.append(x); ys.append(y)
        if b in (320, -1):
            continue
        ax.annotate("160, 320, ∞" if b == 160 else BLAB[b], (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=INK2)
    ax.plot(xs, ys, marker="o", color=BLUE, label="DCOP")
    c = df[df.method == "central"]
    ax.scatter([c.comm.mean()], [c.J.mean() - 0.1 * c.comm.mean()], color=INK, marker="D", s=22,
               label="centralised", zorder=3)
    ax.set_xlabel("messages actually used, $C_{comm}$")
    ax.set_ylabel("$J$ excl. comm. term")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.png"))
    plt.close(fig)


# ------------------------------------------------------------------- tables
def fmt(m, c, nd=2):
    return f"{m:.{nd}f}" if np.isnan(c) else f"{m:.{nd}f}\\,$\\pm$\\,{c:.{nd}f}"


def table_main():
    df = load("main")
    rows = []
    for meth, lab in (("central", "Centralised"), ("dcop", "DCOP (MGM)"), ("dcop-nomgm", "No-MGM")):
        d = df[df.method == meth]
        budgets = [-1] if meth == "central" else [0, 10, 20, 40, 80, -1]
        for b in budgets:
            s = d[d.budget == b]
            r = [lab if b in (-1, 0) and (meth == "central" or b == 0) else "",
                 "--" if meth == "central" else BLAB[b].replace("∞", "$\\infty$")]
            r.append(fmt(*mean_ci(s.success)))
            r.append(fmt(*mean_ci(s.loc[s.success == 1, "T"]), 1))
            r.append(fmt(*mean_ci(s.comm), 1))
            r.append(fmt(*mean_ci(s.move), 0))
            r.append(f"{s.dup_steps.mean():.1f}")
            r.append(f"{s.redundancy.mean():.2f}")
            rows.append(" & ".join(r) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    with open(os.path.join(RES, "table_main.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")


def table_reliability():
    df = load("reliability")
    lines = []
    for rel in ("high", "medium", "low"):
        d = df[df.reliability == rel]
        c = d[d.method == "central"]
        g = lambda b, col: d[(d.method == "dcop") & (d.budget == b)][col].mean()  # noqa: E731
        lines.append(" & ".join([
            rel, f"{c.success.mean():.2f}", f"{g(-1, 'success'):.2f}", f"{g(20, 'success'):.2f}",
            f"{g(0, 'success'):.2f}", f"{c['T'].mean():.1f}", f"{g(-1, 'T'):.1f}", f"{g(20, 'T'):.1f}",
            f"{g(0, 'T'):.1f}", f"{g(-1, 'wrong_claims'):.2f}", f"{g(-1, 'false_trips'):.2f}"]) + " \\\\")
    with open(os.path.join(RES, "table_reliability.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def summary_txt():
    out = []
    for name in ("main", "reliability", "team", "complexity", "range"):
        df = load(name)
        keys = [k for k in ("reliability", "team", "complexity", "comm_range") if k in df]
        g = df.groupby(keys + ["method", "budget"])
        cols = ["success", "T", "comm", "move", "t_detect", "t_handoff", "dup_steps",
                "wrong_claims", "false_trips", "redundancy", "J"]
        out.append(f"== {name} (n={g.size().iloc[0]} seeds per cell)\n{g[cols].mean().round(3).to_string()}\n")
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    fig_main(); fig_reliability(); fig_team_complexity(); fig_range(); fig_tradeoff()
    table_main(); table_reliability(); summary_txt()
    print("figures ->", FIG)
