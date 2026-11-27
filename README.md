# MRAC_Tutorial
Model Reference Adaptive Control tutorials in python

## Running the notebooks and tests

- **Notebooks NB00-NB04** need only the packages in `requirements.txt`.
- **Notebooks NB05-NB13** also import the thesis code (`experiments.mrac`) from the separate thesis
  repository, branch `Claude`, as a Python package.
  - Use a Python environment with an editable install of the thesis checkout
    (`pip install -e <thesis checkout>`), or set `MRAC_THESIS_ROOT`.
  - This repository names no machine path and reads no data files of the thesis repository; see
    `MRAC_RL_NOTEBOOKS.md`.
- **Cached run folders** `results/mrac_*` are local and not in Git. `RESULTS_REGEN.md` lists the
  command that produced each one.
- **Tests:** run `python -m pytest tests` from this folder. Plain `pytest` may not find `mrac_lab`,
  because there is no package configuration. The tests are synthetic and run no episode. `pytest`
  itself is not in `requirements.txt`.
- **Tested environment** (2026-11-27, Windows, CPython 3.12.3):

  | Package | Version |
  |---|---|
  | numpy | 2.5.4 |
  | scipy | 1.18.1 |
  | matplotlib | 3.11.2 |
  | pandas | 3.1.0 |
  | sympy | 1.14.0 |
  | ipywidgets | 8.1.9 |
  | ipympl | 0.10.0 |
  | imageio | 2.38.2 |
  | gymnasium | 1.4.0 |
  | stable-baselines3 | 2.9.2 |
  | torch | 2.15.0+cpu |
  | notebook | 7.6.4 |
  | pytest | 8.4.2 |

  The lower bounds in `requirements.txt` were not tested. `shimmy` is listed there but is not
  imported by any notebook or module, and it was not installed in the tested environment.
