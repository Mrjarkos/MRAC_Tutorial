"""Compute the arm-3 A0 grid once and cache it: ``python -m mrac_lab.make_a0_grid``.

Writes the new file ``--out`` (summary metrics and telemetry, provenance header); it never overwrites the tracked ``results/arm3_a0_grid.csv``.
Scope (approved): 9 conditions x 2 scenarios x 3 compared controllers = 54 episodes plus
18 normaliser episodes; one deterministic run per cell.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True,
                    help="new CSV path; an existing file is refused (results/arm3_a0_grid.csv is tracked historical evidence)")
    ap.add_argument("--thesis-root", default=None)
    args = ap.parse_args()
    from pathlib import Path

    if Path(args.out).exists():  # refuse before any episode runs (save_csv would refuse at the end)
        raise SystemExit(f"--out {args.out} already exists; choose a new path (cached grids are never overwritten)")

    from mrac_lab import a0_grid, provenance

    prov = provenance.setup(args.thesis_root)
    from experiments.mrac.baselines import canonical_offline_params

    canon = canonical_offline_params()
    t0 = time.time()
    df = a0_grid.run_grid(prov.root, workers=args.workers)
    a0_grid.save_csv(df, args.out, prov, canon, args.workers, time.time() - t0)
    print(f"wrote {args.out}: {len(df)} rows in {time.time() - t0:.0f} s; failures {int(df.failed.sum())}")


if __name__ == "__main__":
    main()
