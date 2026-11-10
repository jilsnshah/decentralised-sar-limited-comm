# Decentralised Multi-Agent Search and Rescue under Limited Communication

Final project for **Multi-Agent Systems** (IIT Gandhinagar).
Boinapally Dhanvini (24110079), Bollineni Snehitha Rao (24110081), Jils Shah (24110148), Thaker Shivangi Pareshkumar (24110371).

A person is trapped at an unknown location in a hazardous building. **Search drones** sense well but cannot rescue;
**ground robots** can rescue but sense poorly, so the victim location must travel from drones to robots over a
**limited communication budget** `B`. We compare:

| Approach | Information | Task allocation | Communication |
|---|---|---|---|
| **Centralised baseline** | one pooled Bayesian belief | exact assignment, Hungarian method (`scipy.optimize.linear_sum_assignment`) | counted, never limited |
| **Decentralised DCOP** | each agent's own Bayesian belief | **MGM** (Maximum Gain Message) local search over the range-limited comm graph | range-limited broadcasts, each agent gets `B / |A|` messages |
| **No-MGM ablation** | own belief | independent best responses, no coordination messages | observation messages only |

The full report (8 pages, with Methodology, Results and Contribution sections) is in
[`report/main.pdf`](report/main.pdf); its LaTeX source is [`report/main.tex`](report/main.tex).

## Main results (300 matched seeds per setting, 25,200 episodes)

| Method | B | Success | T_rescue (succ.) | C_comm | C_move | steps violating (2) |
|---|---|---|---|---|---|---|
| Centralised | – | 1.00 | 53.3 | 286.0 | 166 | 0.0 |
| DCOP (MGM) | 0 | 0.74 | 85.5 | 0.0 | 364 | 77.5 |
| DCOP (MGM) | 10 | 0.89 | 68.2 | 9.0 | 288 | 48.4 |
| DCOP (MGM) | 20 | 0.92 | 67.4 | 18.8 | 280 | 44.8 |
| DCOP (MGM) | 40 | 0.94 | 67.0 | 38.2 | 265 | 37.9 |
| DCOP (MGM) | 80 | 0.98 | 60.4 | 74.1 | 240 | 20.0 |
| DCOP (MGM) | ∞ | 1.00 | 58.2 | 125.8 | 225 | 10.4 |
| No-MGM | 20 | 0.95 | 66.4 | 18.2 | 262 | 43.9 |
| No-MGM | ∞ | 0.98 | 66.7 | 77.8 | 278 | 39.1 |

Key findings:

1. **Performance drops sharply only near zero budget.** Two messages per agent (B=10), which carry the drone→robot
   handoff, raise success from 0.74 to 0.89 and halve the time robots wait to learn the victim location (71 → 37 steps).
2. **Decentralised ≈ centralised with less communication.** With unrestricted budget the DCOP team is 9 % slower than the
   centralised coordinator while using 44 % of its messages.
3. **Communication matters more when sensors are noisy** (success gap B=0 vs ∞: 0.20 / 0.26 / 0.39 for high / medium / low
   reliability), but small budgets make robots chase false alarms because claims arrive without the evidence that refutes them.
4. **Coordination messages pay off only with enough budget.** Below ~40 messages MGM's VALUE/GAIN messages crowd out
   observation sharing and the uncoordinated ablation succeeds more often; with large budgets MGM cuts duplicated
   assignments from 39 to 10 steps and completion time from 68 to 58.
5. Smaller teams and larger buildings depend more on communication; widening the comm range closes the gap to the
   centralised baseline only when the budget is not binding.

### Figures

Budget sweep (E1):

![budget sweep](figures/budget_main.png)

Sensor reliability (E2):

![reliability](figures/reliability.png)

Team size and building complexity (E3, E4):

![team and complexity](figures/team_complexity.png)

Communication range (E5) and the communication/performance trade-off:

<p><img src="figures/range.png" width="58%"> <img src="figures/tradeoff.png" width="38%"></p>

### Simulator screenshots (seed 3)

Obstacles black, hazards red, exit **E**, drones = blue triangles, robots = orange squares, victim = star, dashed lines =
agents in communication range, purple = belief of drone d0 (or the coordinator's pooled belief).

Decentralised, unrestricted budget:
![dcop B=inf](figures/snapshots_dcop_Binf_s3.png)

Decentralised, no communication (B = 0): drones find the victim early but cannot tell the robots:
![dcop B=0](figures/snapshots_dcop_B0_s3.png)

Centralised baseline:
![central](figures/snapshots_central_Binf_s3.png)

Animated episode: [`figures/episode_dcop_Binf_s3.gif`](figures/episode_dcop_Binf_s3.gif)

![episode](figures/episode_dcop_Binf_s3.gif)

## Repository layout

```
sar/
  config.py        parameters, sensor-reliability and building-complexity presets
  env.py           building generation (walls, obstacles, hazards, exit, victim), shortest paths
  agent.py         heterogeneous agents, noisy sensor model (Eq. 5), Bayesian belief
  allocation.py    task set, cost functions, Hungarian (centralised) and MGM best response (DCOP)
  sim.py           simulator, messaging rule, budget accounting, metrics
experiments/
  run_experiments.py   all experiments (E1–E5) -> results/*.csv
  make_plots.py        figures/*.png and LaTeX tables
  demo.py              episode snapshots and GIF
tests/test_model.py    checks constraints (2)–(4), the budget (6) and the Bayes update
results/               raw per-episode CSVs, summary.txt, LaTeX tables
figures/               all plots and screenshots
report/                LaTeX source and compiled PDF of the final report
docs/                  project-review report
```

## Reproducing every number

Requires Python ≥ 3.9.

```bash
pip install -r requirements.txt
python tests/test_model.py                     # sanity checks of the formal model
python experiments/run_experiments.py          # 25,200 episodes, ~4 min on 10 cores
python experiments/make_plots.py               # figures/ + results/summary.txt + tables
python experiments/demo.py --seed 3 --budget -1 --gif   # screenshots (-1 = unrestricted)
python experiments/demo.py --seed 3 --budget 0
python experiments/demo.py --seed 3 --method central
cd report && pdflatex main.tex && pdflatex main.tex     # report (copy figures/ into report/figures first)
```

Runs are deterministic: the same seed gives the same building and the same sensor noise stream, and every method/budget
is evaluated on the same seeds. A quick run with fewer seeds: `python experiments/run_experiments.py --seeds 20 --only main`.

## Contribution

| Member | Task | Status |
|---|---|---|
| Boinapally Dhanvini | Literature review, formal model (Section 3), choice of MGM, cost functions | Fulfilled |
| Bollineni Snehitha Rao | Simulator: building, hazards, movement, observation model, belief, reliability/complexity settings | Fulfilled |
| Jils Shah | Centralised Hungarian baseline, decentralised MGM/DCOP, communication budget and accounting, tests | Fulfilled |
| Thaker Shivangi Pareshkumar | Experimental design, metrics, plots, tables and analysis | Fulfilled |

All members contributed to integration, testing, debugging and the report.
