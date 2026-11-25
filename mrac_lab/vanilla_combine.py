"""Combine a wall-capped stage with its completion run into one composite folder.

``python -m mrac_lab.vanilla_combine <manifest.json> <part1> <part2> [...]`` writes
``results/<stage>_composite_<stamp>/`` with the full manifest, the union of results (a
``cancelled_wall_cap`` result is replaced by the later part's result for the same id)
and a receipt that names every part, its status and its source hashes. It refuses
unless every manifest id has exactly one final result that is neither cancelled nor
errored. Parts are never modified (protocol 06 AM7).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    parts = [Path(p) for p in sys.argv[2:]]
    final: dict[str, dict] = {}
    receipts = []
    for part in parts:
        receipts.append({"folder": str(part.name), **json.loads((part / "receipt.json").read_text())})
        for r in json.loads((part / "summary.json").read_text()):
            if r.get("status") in ("cancelled_wall_cap", "error"):
                continue
            if r["id"] in final:
                raise SystemExit(f"duplicate final result for {r['id']}")
            final[r["id"]] = {**r, "source_folder": part.name}
    ids = [j["id"] for j in manifest]
    missing = [i for i in ids if i not in final]
    if missing or set(final) - set(ids):
        raise SystemExit(f"incomplete composite: missing {missing[:5]} ({len(missing)}), extra {sorted(set(final) - set(ids))[:5]}")
    stage = receipts[0]["stage"]
    out = ROOT / "results" / f"mrac_vanilla_{stage}_composite_{time.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "jobs.json", "x") as fh:
        json.dump(manifest, fh, indent=1)
    with open(out / "summary.json", "x") as fh:
        json.dump([final[i] for i in ids], fh, indent=1, default=str)
    receipt = {"stage": stage, "status": "completed", "composite_of": receipts,
               "rule": "protocol 06 AM7: cancelled_wall_cap results replaced by the completion run",
               "excluded_designs": [], "excluded_jobs": []}
    with open(out / "receipt.json", "x") as fh:
        json.dump(receipt, fh, indent=1, default=str)
    print(f"wrote {out}: {len(ids)} results from {len(parts)} parts")


if __name__ == "__main__":
    main()
