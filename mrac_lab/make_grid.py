"""Compute the offline S30-style grid once and cache it: ``python -m mrac_lab.make_grid``.

Writes the new file ``--out`` (summary metrics, provenance header); it never overwrites the tracked ``results/s30_offline_grid.csv``.
Scope (approved): 25 conditions x 4 arms x 2 scenarios = 200 deterministic episodes.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True,
                    help="new CSV path; an existing file is refused (results/s30_offline_grid.csv is tracked historical evidence)")
    ap.add_argument("--thesis-root", default=None)
    args = ap.parse_args()
    from pathlib import Path

    if Path(args.out).exists():  # refuse before any episode runs (save_csv would refuse at the end)
        raise SystemExit(f"--out {args.out} already exists; choose a new path (cached grids are never overwritten)")

    from mrac_lab import grid, provenance

    prov = provenance.setup(args.thesis_root)
    from experiments.mrac.baselines import canonical_offline_params, s30_conditions

    canon = canonical_offline_params()
    t0 = time.time()
    df = grid.run_grid(prov.root, s30_conditions(), workers=args.workers, fixed_form=True)
    grid.save_csv(df, args.out, prov, canon, args.workers, time.time() - t0)
    print(f"wrote {args.out}: {len(df)} rows in {time.time() - t0:.0f} s; failures {int(df.failed.sum())}")


if __name__ == "__main__":
    main()
