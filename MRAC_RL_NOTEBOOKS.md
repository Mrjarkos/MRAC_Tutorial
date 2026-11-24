# MRAC-RL slung-load notebooks (05-07): quick guide

These three notebooks are an **exploratory logbook**. Results are offline (a reduced Newton-Euler model of the quadrotor and a taut point load), run on the development PC (PROXY). They are not Pymunk results, not a stability proof and not a study.

## Where things live

| What | Where |
|---|---|
| Notebooks, plotting and widget helpers | this repo, `main`: `05_*.ipynb`, `06_*.ipynb`, `07_*.ipynb`, `mrac_lab/`, `results/s30_offline_grid.csv` |
| Physics, controllers, metrics (tested code) | thesis repo `gym-pymunk-quadrotor-suspended-load`, branch **`Claude`**, folder `experiments/mrac/` (`quad.py`, `quad_load_pend.py`, `law.py`, `payload_law.py`, `baselines.py`, `offline_metrics.py`) with tests in `tests/mrac/` |
| The mathematics | thesis repo `.ai/research/derivations/mrac-quadrotor-payload.md` (cited as "MQP §n" in the notebooks) |

The notebooks import the thesis code from the main thesis checkout. That is the folder `...\Personal\thesis\gym-pymunk-quadrotor-suspended-load`, which must be on the `Claude` branch.
- **A different checkout:** to use another thesis checkout, set `MRAC_THESIS_ROOT` before starting Jupyter.
- **Provenance banner:** the first code cell prints the commit, branch and parameter sources, plus a warning if the checkout moved past the commit the notebook was written against.

**Refactor worktrees are unrelated.** The other thesis folders and worktrees (`p7w1-*`, `p7w2-*`, `phase6/*`, `.ai/work/...`) belong to the Phase 6/7 refactor. These notebooks do not use them.

## How to launch

1. Open this folder in VS Code.
2. Open a notebook and select the kernel **Python (`.venv-workspace`)**, the interpreter at `...\thesis\gym-pymunk-quadrotor-suspended-load\.venv-workspace\Scripts\python.exe`. Not this repo's `.venv`: it lacks sympy and pandas.
3. Run All. Approximate times:

   | Notebook | Time |
   |---|---|
   | NB05 | about 2.5 min |
   | NB06 | about 1.5 min (it loads the cached grid; set `RECOMPUTE = True` only to rerun the 250 offline episodes, about 4 min on 6 cores) |
   | NB07 | about 1.5 min |

4. The interactive panels (sliders and dropdowns) redraw inline figures. Change a value and wait for the redraw.

Command line alternative: `python -m jupyter nbconvert --to notebook --execute --inplace 06_Baselines_PIDFF_and_Linear_Damper.ipynb`, using the `.venv-workspace` Python.

## What to show (about 20 minutes)

| Step | Notebook | Show | Key numbers |
|---|---|---|---|
| 1 | NB05 | Problem statement, model, validations; the slider for `m_L`, `l`, `d`; Fact 1 (same vehicle motion gives the same swing for any mass, a different swing for another length) | hover residual 1.4e-14; energy rate 5e-15; free-vehicle frequency ratio `sqrt(1 + m_L/m_q)` to 2.5e-9 |
| 2 | NB06 (a) | Arm 1: PID+FF with a mass-blind FF (assumes 4 kg); sag with heavier loads | the FF gate uses ideal tension offline (S30 used filtered tension) |
| 3 | NB06 (b) | Linear-damper derivation: damper only in the attitude row; `N(s)` has no controller gain; zero pair `+-sqrt(g(m+m_L)/(d m))`, about 6 rad/s, from the **offset attachment** | least-damped `zeta`: PID+FF 0.158, softer law 0.244, damper 0.378, peak 0.419 at `kd` about -1.7; not monotone; continuous-time limit near `abs(kd)` 10 (also exists at `d = 0`) |
| 4 | NB06 (c-d) | Arm 2 and the offline S30 grid heatmaps | 25 conditions, 5 controllers, 250 runs, 0 failures; ratios vs true-mass PID+FF; only the softer law with true-mass FF is like-for-like with S30 |
| 5 | NB07 | Arm 3 design: the law, the `Q`-gate check, the open obligations; the preliminary run as "pipeline works" | payload estimates did not order with mass; ONLINE is worse than arm 1; the reference with a nominal payload is the next derivation |

**Do not say:**
- That the damper is stable up to `abs(kd) = 10` in the real or sampled loop.
- That the zero causes that limit, or that the cable torque causes the zero.
- That damping rises monotonically.
- That the offline grid reproduces S30.
- That MRAC improves tracking or identifies the mass.
- That the `Q` gate proves stability.

## Prepare now (about 15 minutes)

1. Run all three notebooks once with the kernel above and check that the banner shows branch `Claude`.
2. In NB06, read §4 (derivation) and §8 (interpretation); move the sliders once.
3. In NB07, read §4.5 (open obligations, including obligation 6) and §8.
4. Optional reading in the thesis repo: `.ai/work/2026-11-24-mrac-notebooks/05_meeting_brief.md` (talking points) and `04_notebook_math_check.md` (independent check of the mathematics).

## Extending the work

- **Code:** new physics, controllers or metrics go into the thesis repo's `experiments/mrac/`, with tests, on `Claude` (one commit on top, or a short branch fast-forwarded into `Claude`).
- **Notebooks:** new notebooks go here, numbered 08 onward, importing that code. Keep each notebook's sections: question, hypotheses, provenance, theory, exploration, observations, table, interpretation, next steps.
