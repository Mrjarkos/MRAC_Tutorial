"""Where the tested offline code lives, which commit it is, and a printed banner.

The notebooks never copy physics. They import ``experiments.mrac`` (and
``experiments.notebook_utils.style``) from the thesis repository. This module
makes that import explicit and reproducible:

* the thesis root is configurable (argument, or the ``MRAC_THESIS_ROOT``
  environment variable, or the default below);
* ``setup`` prepends it to ``sys.path``, verifies that the imported modules
  really come from it, and reads the commit with ``git``;
* ``banner`` prints the commit, the parameter sources and the
  EXPLORATORY / offline / PROXY label.

Nothing here installs, trains, or touches the network.
"""

from __future__ import annotations

import html
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

#: Default thesis checkout: the main checkout on the ``Claude`` branch.
DEFAULT_THESIS_ROOT = Path(
    r"C:\Users\Jcardenasbohorquez\OneDrive - Expleo France\Dokumente\Personal\thesis"
    r"\gym-pymunk-quadrotor-suspended-load"
)
ENV_VAR = "MRAC_THESIS_ROOT"

#: Paths whose uncommitted state can change notebook results (2026-11-25: scoped to the
#: imported code; unrelated files elsewhere under ``experiments/`` no longer mark a run dirty).
DIRTY_SCOPE = ("experiments/mrac", "experiments/notebook_utils/style.py", "gym_pymunk_quadrotor")

#: Label printed on every notebook. Offline results are PROXY results.
LABEL = "EXPLORATORY / offline / PROXY (development PC, reduced taut point-load model)"
CLAIM_LIMITS = (
    "No stability claim, no Pymunk-transfer claim, no hardware claim. "
    "Offline metrics are named *_off and are not the canonical c_T / c_S."
)


@dataclass(frozen=True)
class Provenance:
    """Resolved thesis checkout used by this kernel."""

    root: Path
    commit: str
    branch: str
    dirty: bool
    python: str
    origins: Mapping[str, str] = field(default_factory=dict)
    dirty_paths: Sequence[str] = ()

    @property
    def short_commit(self) -> str:
        return self.commit[:7]


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def resolve_root(thesis_root: Optional[os.PathLike] = None) -> Path:
    """Argument, else ``MRAC_THESIS_ROOT``, else the default W2c worktree."""
    raw = thesis_root or os.environ.get(ENV_VAR) or DEFAULT_THESIS_ROOT
    root = Path(raw).expanduser().resolve()
    if not (root / "experiments" / "mrac").is_dir():
        raise FileNotFoundError(
            f"{root} has no experiments/mrac; set {ENV_VAR} or pass thesis_root."
        )
    return root


def setup(
    thesis_root: Optional[os.PathLike] = None,
    pinned_commit: Optional[str] = None,
    modules: Sequence[str] = (
        "experiments.mrac.quad",
        "experiments.mrac.quad_load_pend",
        "experiments.notebook_utils.style",
    ),
) -> Provenance:
    """Put the thesis root first on ``sys.path`` and record where code comes from.

    ``pinned_commit`` (a SHA prefix) is optional: if given and the checkout is
    at another commit, a warning is printed, not raised, so a reader can still
    run the notebook on a newer checkout.
    """
    root = resolve_root(thesis_root)
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    importlib.invalidate_caches()

    origins: dict[str, str] = {}
    for name in modules:
        mod = importlib.import_module(name)
        file = Path(mod.__file__ or "").resolve()
        origins[name] = str(file)
        if root not in file.parents:
            raise ImportError(
                f"{name} was imported from {file}, not from the thesis root {root}. "
                "Restart the kernel (another checkout was imported first)."
            )

    commit = _git(root, "rev-parse", "HEAD") or "UNKNOWN"
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "UNKNOWN"
    status = _git(root, "status", "--porcelain", "--", *DIRTY_SCOPE)
    # porcelain lines are "XY path"; _git strips the leading space of the first line
    dirty_paths = tuple(line.split(None, 1)[1] for line in status.splitlines() if line.strip())
    prov = Provenance(
        root, commit, branch, bool(dirty_paths), sys.version.split()[0], origins, dirty_paths
    )
    if pinned_commit and not commit.startswith(pinned_commit):
        print(
            f"WARNING: checkout is at {commit[:7]}, notebook was written against "
            f"{pinned_commit[:7]}. Results may differ."
        )
    return prov


def banner(
    prov: Provenance,
    title: str,
    sources: Optional[Mapping[str, str]] = None,
    notes: Sequence[str] = (),
) -> None:
    """Print and display the experiment-identity banner (notebook section 3)."""
    lines = [
        f"{title}",
        f"Label: {LABEL}",
        f"Thesis checkout: {prov.root}",
        f"Commit: {prov.commit} (branch {prov.branch}"
        + (", UNCOMMITTED CHANGES in " + ", ".join(prov.dirty_paths) if prov.dirty else ", clean for " + ", ".join(DIRTY_SCOPE))
        + ")",
        f"Python: {prov.python}",
        f"Claim limits: {CLAIM_LIMITS}",
    ]
    for name, path in prov.origins.items():
        lines.append(f"import origin: {name} -> {path}")
    if sources:
        lines.append("Parameter sources:")
        lines.extend(f"  {k}: {v}" for k, v in sources.items())
    lines.extend(f"Note: {n}" for n in notes)
    print("\n".join(lines))

    try:  # coloured box for screen readers of the notebook; text above is the record
        from IPython.display import HTML, display

        rows = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
        src_rows = "".join(
            f"<tr><td style='padding:0 12px 0 0'><code>{html.escape(k)}</code></td>"
            f"<td>{html.escape(v)}</td></tr>"
            for k, v in (sources or {}).items()
        )
        display(
            HTML(
                "<div style='border-left:6px solid #D55E00;background:#FFF4EC;"
                "padding:8px 14px;margin:6px 0;font-size:90%'>"
                f"<b>{html.escape(title)}</b><br>"
                f"<b>{html.escape(LABEL)}</b><br>"
                f"commit <code>{prov.short_commit}</code> on <code>{html.escape(prov.branch)}</code>"
                + (" <b>(uncommitted changes)</b>" if prov.dirty else "")
                + f"<br>{html.escape(CLAIM_LIMITS)}"
                + (f"<table style='margin-top:6px'>{src_rows}</table>" if src_rows else "")
                + (f"<ul style='margin:4px 0 0 0'>{rows}</ul>" if rows else "")
                + "</div>"
            )
        )
    except Exception:  # pragma: no cover - display is cosmetic
        pass
