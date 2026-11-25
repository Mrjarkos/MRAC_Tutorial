"""Protocol 08 section 6 comparisons for the extension stages (descriptive, deterministic).

``python -m mrac_lab.ext_analysis <folder_a> [<folder_ab> <folder_abc>]`` writes
``ext_analysis.json`` next to the last folder (exclusive create) and prints the tables.

Rule (frozen in protocol 08): "X improves on Y in disturbance class D" when the paired
difference of plant-to-task RMS (last 60 s), X minus Y, is negative in >= 75 % of D's
comparable cells AND G3' (swing 10-30 s <= max(B_S, 0.005 rad), B_S from softer / pid_ff
in the same cell) is True for X there. Scored symmetrically in both directions; failed
episodes make a cell non-comparable (never an opponent win); unknown G3' never passes;
only the 'grid' and 'dist' sets are scored (Codex 27 EXT-01..04).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

FLOOR = 0.005
CLASSES = {"none": "D0 none", "const": "D1 constant", "ou": "D2 OU", "drag": "D3 drag", "gust": "D4 gust",
           "payload_kick": "D5 payload kick", "vehicle_kick": "D6 vehicle kick"}
PAIRS = [("a_online", "v0_online"), ("ab_online", "a_online"), ("abc_online", "ab_online"),
         ("a_online", "a_frozen"), ("ab_online", "ab_frozen"), ("abc_online", "abc_frozen"),
         ("a_online", "softer"), ("ab_online", "softer"), ("abc_online", "softer"),
         ("a_online", "pidff"), ("v0_online", "softer")]


def load(folders) -> pd.DataFrame:
    frames = []
    for f in folders:
        rec = json.loads((Path(f) / "receipt.json").read_text())
        if rec.get("status") != "completed":
            raise SystemExit(f"{f}: receipt status {rec.get('status')!r}; refusing partial data")
        df = pd.DataFrame(json.loads((Path(f) / "summary.json").read_text()))
        df["folder"] = str(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    df["cell"] = df.label + "|" + df.scenario + "|" + df.dist + df.seed.astype(str) + "|" + df.tag + "|d" + df.d.astype(str)
    return df


def _ok(r) -> bool:
    return r is not None and r.get("status") == "completed" and math.isfinite(r.get("task_rms_last60", math.nan))


SCORED_TAGS = ("grid", "dist")  # protocol 08: 'long' and 'robust' are readouts, not scored


def g3(by: dict, cell: str, arm: str):
    """True / False, or None when not evaluable (a baseline or the arm itself is missing or failed)."""
    r, soft, pid = by.get((cell, arm)), by.get((cell, "softer")), by.get((cell, "pidff"))
    if not (_ok(r) and _ok(soft) and _ok(pid)):
        return None
    bs = soft["swing_rms_10_30"] + 0.5 * max(pid["swing_rms_10_30"] - soft["swing_rms_10_30"], 0.0)
    return bool(r["swing_rms_10_30"] <= max(bs, FLOOR))


def compare(df: pd.DataFrame) -> dict:
    """Symmetric scoring (Codex 27 EXT-01..03).

    Per cell: comparable only when both episodes completed with finite metrics (a failed
    episode is a failed-arm outcome, never an opponent win). A side scores when its task
    error is lower AND its own G3' is True (None does not pass). Verdict per direction:
    scores >= 75 % of the comparable cells. Unknown-G3' and failed counts are reported.
    """
    df = df[df.tag.isin(SCORED_TAGS)]
    by = {(r.cell, r.arm): r._asdict() for r in df.itertuples()}
    cells = sorted(df.cell.unique())
    out = {}
    for x, y in PAIRS:
        per = {}
        for c in cells:
            rx, ry = by.get((c, x)), by.get((c, y))
            if rx is None or ry is None:
                continue
            dist = c.split("|")[2].rstrip("0123456789")
            d = per.setdefault(dist, {"n_cells": 0, "comparable": 0, "failed_x": 0, "failed_y": 0,
                                      "x_lower": 0, "y_lower": 0, "x_scores": 0, "y_scores": 0,
                                      "g3_unknown_x": 0, "g3_unknown_y": 0, "xs": [], "ys": [], "y_score_cells": []})
            d["n_cells"] += 1
            d["failed_x"] += not _ok(rx)
            d["failed_y"] += not _ok(ry)
            if not (_ok(rx) and _ok(ry)):
                continue
            d["comparable"] += 1
            gx, gy = g3(by, c, x), g3(by, c, y)
            d["g3_unknown_x"] += gx is None
            d["g3_unknown_y"] += gy is None
            tx, ty = rx["task_rms_last60"], ry["task_rms_last60"]
            d["xs"].append(tx); d["ys"].append(ty)
            if tx < ty:
                d["x_lower"] += 1
                d["x_scores"] += gx is True
            elif ty < tx:
                d["y_lower"] += 1
                d["y_scores"] += gy is True
                if gy is True:
                    d["y_score_cells"].append(c)
        res = {}
        for dist, d in per.items():
            n = d["comparable"]
            res[CLASSES.get(dist, dist)] = {
                **{k: v for k, v in d.items() if k not in ("xs", "ys", "y_score_cells")},
                "x_improves": n > 0 and d["x_scores"] >= 0.75 * n,
                "y_improves": n > 0 and d["y_scores"] >= 0.75 * n,
                "median_x": float(pd.Series(d["xs"]).median()) if d["xs"] else float("nan"),
                "median_y": float(pd.Series(d["ys"]).median()) if d["ys"] else float("nan"),
                "y_score_cells": d["y_score_cells"][:12],
            }
        out[f"{x} vs {y}"] = res
    return out


def kick_table(df: pd.DataFrame) -> pd.DataFrame:
    k = df[df.dist.isin(["payload_kick", "vehicle_kick"]) & (df.tag == "dist")]
    cols = [c for c in ("kick0_swing_rms_5s", "kick0_task_max_5s", "kick0_settle_s", "kick1_settle_s") if c in k]
    return k.groupby(["dist", "arm"])[cols].median()


def long_table(df: pd.DataFrame) -> pd.DataFrame:
    lg = df[df.tag == "long"]
    cols = [c for c in lg.columns if c.startswith(("k_t_", "k_u_", "k_phi_")) or c in ("task_rms_last60", "status")]
    return lg.set_index(["label", "arm"])[sorted(cols)]


def main() -> None:
    folders = [Path(f) for f in sys.argv[1:]]
    df = load(folders)
    res = {"folders": [str(f) for f in folders], "rule": __doc__.split("Rule")[1].strip(), "comparisons": compare(df),
           "n_episodes": int(len(df)), "n_failed": int((df.status != "completed").sum()),
           "failed": df[df.status != "completed"][["id", "status", "stop"]].to_dict("records")}
    import os

    target = folders[-1] / os.environ.get("EXT_ANALYSIS_NAME", "ext_analysis.json")
    with open(target, "x") as fh:
        json.dump(res, fh, indent=1, default=str)
    for pair, r in res["comparisons"].items():
        line = "; ".join(f"{d}: X {v['x_scores']}/{v['comparable']}{'*' if v['x_improves'] else ''}"
                         f" Y {v['y_scores']}{'*' if v['y_improves'] else ''} (g3? {v['g3_unknown_x']})" for d, v in r.items())
        print(f"{pair:24s} {line}")
    print(kick_table(df).round(4))
    print(f"episodes {res['n_episodes']}, failed {res['n_failed']}")


if __name__ == "__main__":
    main()
