"""Preflight gate for the hover-Jacobian rows (protocol 08 AM5, Javier 2026-11-27).

One pure rule shared by ``vanilla_runs`` and ``extensions_runs``. No thesis import.

A row is the per-plant result of ``hover_tick_jacobian`` / ``nl_hover_jacobian``:
``rate_per_s`` (reduced-map growth rate), ``invariance`` (``||Q_R^T J S||``) and
``drift`` (``max |f(z0) - z0|``). The status is:

- ``fail``: the rate is not finite or above the limit, or ``S`` is not invariant
  (``invariance >= 1e-6``). The neutral subspace holds the collinear RBF split only when
  ``sigma_W == sigma``, so ``S`` is exactly invariant for every variant and the strict
  tolerance applies to all of them.
- ``not_certified``: the rate and invariance pass, but the tick is not at an equilibrium
  (``drift > 1e-9``, e.g. motor asymmetry: hover is not a fixed point). The rate is then
  reported, not certified, and does not stop a run. A plant the caller marks as having no
  exact ideal (``ideal_equilibrium=False``, vanilla ``l != l_r``) is never gated: it is
  ``not_certified`` whatever its values, with the would-be failures listed as reasons.
- ``pass``: otherwise.
"""

from __future__ import annotations

import math

INVARIANCE_TOL = 1e-6
DRIFT_TOL = 1e-9
NOT_FIXED_POINT = "not a fixed point, not certified"


def classify(
    row: dict,
    *,
    rate_limit: float,
    inclusive: bool,
    ideal_equilibrium: bool = True,
) -> dict:
    """``{"status": "pass" | "fail" | "not_certified", "reasons": [...]}`` for one row.

    The rate passes when ``rate <= rate_limit`` (``inclusive``) or ``rate < rate_limit``.
    """
    rate = float(row["rate_per_s"])
    fails, notes = [], []
    if not math.isfinite(rate):
        fails.append("rate not finite")
    elif not (rate <= rate_limit if inclusive else rate < rate_limit):
        fails.append(f"rate {rate:.3g}/s {'>' if inclusive else '>='} {rate_limit:g}/s")
    inv = float(row["invariance"])
    if not (math.isfinite(inv) and inv < INVARIANCE_TOL):
        fails.append(f"invariance {inv:.3g} >= {INVARIANCE_TOL:g}")
    drift = float(row["drift"])
    if not math.isfinite(drift):
        fails.append("drift not finite")
    elif drift > DRIFT_TOL:
        notes.append(f"drift {drift:.3g}: {NOT_FIXED_POINT}")
    if not ideal_equilibrium:
        return {"status": "not_certified",
                "reasons": ["no exact ideal at this plant: not gated"] + fails + notes}
    if fails:
        return {"status": "fail", "reasons": fails + notes}
    if notes:
        return {"status": "not_certified", "reasons": notes}
    return {"status": "pass", "reasons": []}


__all__ = ["DRIFT_TOL", "INVARIANCE_TOL", "NOT_FIXED_POINT", "classify"]
