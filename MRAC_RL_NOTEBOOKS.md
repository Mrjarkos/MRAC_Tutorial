# MRAC tutorial and slung-load notebooks (00-13): quick guide

NB01-NB04 are generic MRAC tutorials (theory). NB05-NB13 are an **exploratory logbook**. Their results are offline (a reduced Newton-Euler model of the quadrotor and a taut point load), run on the development PC (PROXY). They are not Pymunk results, not a stability proof and not a study.

**Start with NB00** (`00_Notation_and_Reading_Guide.ipynb`): it holds the glossary of every abbreviation used in the series (MQP, EQ-MR-xx, A0, B3, route (b), W1 to W2e, arms 1 to 3, PE, UUB and others), the notation map between thesis `residual.tex`, the simulator code and the Guha-Annaswamy paper, and one complete scalar MRAC derivation that the other notebooks follow.

## Reading order

| Notebook | Question | Needs first |
|---|---|---|
| NB00 Notation and reading guide | What do the symbols and abbreviations mean? | nothing |
| NB01 Introduction to MRAC | How can gains adapt online so a plant with unknown parameters follows a reference model (MIT rule)? | NB00 |
| NB02 MRAC meets Lyapunov | How does a Lyapunov function replace the MIT rule, with a matched disturbance and a chosen basis? | NB01 |
| NB03 Lyapunov, general system | The same for vectors and matrices, with unknown control effectiveness | NB02 |
| NB04 RL and MRAC | How does the MRAC-RL framework (Guha and Annaswamy) use an RL policy's closed loop as the reference model? | NB02, NB03 |
| NB05 Slung-load model validation | What is the taut point-load model, and does it pass its checks? ($d = 0.3795$ m, code angle) | NB00 |
| NB06 Baselines | PID+FF (arm 1), linear damper (arm 2), the torque-to-swing zero pair | NB05 |
| NB08 Vehicle mismatch grid | How much do vehicle parameter errors hurt arms 1 and 2? | NB06 |
| NB07 Arm 3 | The A0 / B3 route-(b) adaptive law | NB06 |
| NB09 Nominal-load reference | A0 with the nominal-payload reference; long-horizon issues | NB07 |
| NB10 Vanilla MRAC (Guha) | The Guha-Annaswamy pattern on the hover-linearised outer model ($d = 0$) | NB04, NB05 |
| NB11 Extension (a) | Nonlinear taut model in cable-frame channels | NB10 |
| NB12 Extension (a)+(b) | Disturbance-estimator weights | NB11 |
| NB13 Extension (a)+(b)+(c) | $\sigma$ leak | NB12 |

The table is in reading order. In the logbook branch, NB05, NB06, NB08 and NB07 are steps 1 to 4 and NB09 follows NB07: NB08 comes before NB07 because it was added after the 2026-11-24 meeting as step 3 of 4.

## What "MQP" means

**MQP** is the short name of the thesis derivation note *MRAC for a planar quadrotor with a cable-suspended payload* (**M**RAC, **Q**uadrotor, **P**ayload): the file `.ai/research/derivations/mrac-quadrotor-payload.md` in the thesis repository. Its status is **PROPOSED**: it is not accepted. "MQP §6.14" means section 6.14 of that file. The notebooks cite it as the reference source and carry the key steps themselves, so a reader does not need the file. The thesis equation registry rows "EQ-MR-xx" point to it.

## Where things live

| What | Where |
|---|---|
| Notebooks, plotting and widget helpers | this repo, `main`: `00_*.ipynb` to `13_*.ipynb`, `mrac_lab/` (including `vanilla_runs.py` and `vanilla_analysis.py`), and the cached grids `results/s30_offline_grid.csv`, `results/vehicle_mismatch_grid.csv`, `results/arm3_a0_grid.csv` |
| Physics, controllers, metrics (tested code) | thesis repo `gym-pymunk-quadrotor-suspended-load`, branch **`Claude`**, folder `experiments/mrac/` (`quad.py`, `quad_load_pend.py`, `law.py`, `payload_law.py`, `baselines.py`, `offline_metrics.py`, `payload_reference.py`, `diagnostics.py`, `wind.py`, `vanilla_mrac.py`) with tests in `tests/mrac/` |
| The mathematics | thesis repo `.ai/research/derivations/mrac-quadrotor-payload.md` = **MQP**, the derivation note *MRAC for a planar quadrotor with a cable-suspended payload* (PROPOSED, not accepted); cited as "MQP §n" in the notebooks, which carry the key steps themselves |

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
   | NB00 | a few seconds (one SymPy cell; the notebook is mostly text) |
   | NB05 | about 2.5 min |
   | NB06 | about 1.5 min (it loads the cached grid; set `RECOMPUTE = True` only to rerun the 250 offline episodes, about 4 min on 6 cores) |
   | NB07 | about 1.5 min (loads the cached arm-3 grid) |
   | NB08 | under 1 min (loads the cached mismatch grid) |
   | NB09 | under 1 min (reads saved A0 long-horizon, stabilization and wind results) |
   | NB10 | about 1 min (vanilla MRAC: closed forms, the hover-Jacobian scan, saved validation and tuning results) |
   | NB11-NB13 | about 1-2 min each (extension series a / a+b / a+b+c: derivation, block diagram, cached protocol-08 grids; interactive panels; the live panel runs one episode only when its button is pressed) |

   To regenerate a grid: `python -m mrac_lab.make_grid`, `python -m mrac_lab.make_mismatch_grid` or `python -m mrac_lab.make_a0_grid`, from this folder, with the `.venv-workspace` Python.

4. The interactive panels (sliders and dropdowns) redraw inline figures. Change a value and wait for the redraw.

Command line alternative: `python -m jupyter nbconvert --to notebook --execute --inplace 06_Baselines_PIDFF_and_Linear_Damper.ipynb`, using the `.venv-workspace` Python.

## What to show (about 20 minutes)

| Step | Notebook | Show | Key numbers |
|---|---|---|---|
| 1 | NB05 | Problem statement, model, validations; the slider for `m_L`, `l`, `d`; Fact 1 (same vehicle motion gives the same swing for any mass, a different swing for another length) | hover residual 1.4e-14; energy rate 5e-15; free-vehicle frequency ratio `sqrt(1 + m_L/m_q)` to 2.5e-9 |
| 2 | NB06 (a) | Arm 1: PID+FF with a mass-blind FF (assumes 4 kg); sag with heavier loads | the FF gate uses ideal tension offline (S30 used filtered tension) |
| 3 | NB06 (b) | Linear-damper derivation: damper only in the attitude row; `N(s)` has no controller gain; zero pair `+-sqrt(g(m+m_L)/(d m))`, about 6 rad/s, from the **offset attachment** | least-damped `zeta`: PID+FF 0.158, softer law 0.244, damper 0.378, peak 0.419 at `kd` about -1.7; not monotone; continuous-time limit near `abs(kd)` 10 (also exists at `d = 0`) |
| 4 | NB06 (c-d) | Arm 2 and the offline S30 grid heatmaps | 25 conditions, 5 controllers, 250 runs, 0 failures; ratios vs true-mass PID+FF; only the softer law with true-mass FF is like-for-like with S30 |
| 5 | NB06 §4.4-4.5 | The non-minimum-phase zero from the attachment offset; oracle vs fixed form | zero 5.59 / 6.05 / 6.48 rad/s at 2 / 4 / 6 kg (`d = 0.38`); bandwidth rule NOT VERIFIED |
| 6 | NB08 | Vehicle mismatch (mass, inertia, arm, motor asymmetry) for PID+FF and the swing laws | asymmetry hurts most (7.5x at 10 %), then vehicle mass (up to 3.3x); inertia and arm at most +4 %; swing laws keep reducing swing |
| 7 | NB07 §10 | Arm 3 with the A0 nominal-payload reference (4 kg / 2 m, softer law inside the reference): ONLINE vs FROZEN (nominal ideals) vs arm 2 | exact identity at nominal; ONLINE 2-5x lower position error than FROZEN; beats arm 2 only for heavier loads; light-load estimate pins at 0; swing damping not inherited (the A0 blind spot `phi - phi_r`); length mismatch: A0 worse than arm 2 |
| 8 | NB09 §8-10 | Why the A0 law fails at long horizon (C1 filter phase margin, C2 measured-swing cancellation, C3 lumped-mass mode) and the EQ-MR-18..21 ablations | PROPOSED, exploratory |
| 9 | NB10 | Vanilla MRAC + linear damper in Guha notation (EQ-MR-22): matching proof, Lyapunov `V' = -e'Qe`, sampled-data hover check, validation, tuning | exact only on the linearized matched outer model; no stability claim for the nonlinear sampled loop |
| 10 | NB11-NB13 | Extensions at `d = 0`: (a) nonlinear model in the cable frame (EQ-MR-23), (b) RBF disturbance weights, (c) σ-modification; vehicle-mismatch grid, wind, drag, gusts, payload and vehicle kicks, 600 s holds; which statement improves which | the derivative identity is proved; local convergence is conditional; protocol 08 is descriptive |

**Do not say:**
- That the damper is stable up to `abs(kd) = 10` in the real or sampled loop.
- That the zero causes that limit, or that the cable torque causes the zero.
- That damping rises monotonically.
- That the offline grid reproduces S30.
- That MRAC improves tracking in general (it helps only for some cells), or that it identifies the mass.
- That the reference's swing damping carries over to the real load.
- That the `Q` gate proves stability.
- That the vanilla law's Lyapunov result covers the nonlinear sampled loop, or that its gains converge to the true parameters.

## Prepare now (about 15 minutes)

1. Run all three notebooks once with the kernel above and check that the banner shows branch `Claude`.
2. In NB06, read §4 (derivation) and §8 (interpretation); move the sliders once.
3. In NB07, read §4.5 (open obligations, including obligation 6) and §8.
4. Optional reading in the thesis repo: `.ai/work/2026-11-24-mrac-notebooks/05_meeting_brief.md` (talking points) and `04_notebook_math_check.md` (independent check of the mathematics).

## Extending the work

- **Code:** new physics, controllers or metrics go into the thesis repo's `experiments/mrac/`, with tests, on `Claude` (one commit on top, or a short branch fast-forwarded into `Claude`).
- **Notebooks:** new notebooks go here, numbered 08 onward, importing that code. Keep each notebook's sections: question, hypotheses, provenance, theory, exploration, observations, table, interpretation, next steps.
