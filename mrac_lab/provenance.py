"""Where the tested offline code lives, which commit it is, and a printed banner.

The notebooks never copy physics. They import ``experiments.mrac`` (and
``experiments.notebook_utils.style``) from the thesis repository. This module
makes that import explicit and reproducible:

* the thesis root is an explicit argument, or the ``MRAC_THESIS_ROOT``
  environment variable, or else the checkout that provides the importable
  ``experiments`` package. The thesis wheel packages only ``gym_pymunk_quadrotor``,
  so ``experiments`` is importable only from an editable install of the checkout
  (``pip install -e <thesis checkout>`` puts its root on ``sys.path``) or from
  ``PYTHONPATH``. No
  machine path is hard-coded, and nothing assumes where the two repositories sit
  relative to each other;
* ``setup`` prepends it to ``sys.path``, verifies that the imported modules
  really come from it, and reads the commit with ``git``;
* ``banner`` prints the commit, how the checkout was found, the import origins
  relative to the checkout, the parameter sources and the
  EXPLORATORY / offline / PROXY label. It prints no absolute path.

Nothing here installs, trains, or touches the network.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

ENV_VAR = "MRAC_THESIS_ROOT"
#: Package whose installed location identifies the thesis checkout.
THESIS_PACKAGE = "experiments"

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
    #: How the checkout was found: "argument", "MRAC_THESIS_ROOT" or "importable experiments package".
    source: str = "argument"

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


def _installed_root() -> Optional[Path]:
    """Checkout that provides the importable ``experiments`` package, if any."""
    try:
        spec = importlib.util.find_spec(THESIS_PACKAGE)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin not in ("namespace", "built-in"):
        return Path(spec.origin).resolve().parents[1]
    locations = list(spec.submodule_search_locations or ())
    return Path(locations[0]).resolve().parent if locations else None


def find_root(thesis_root: Optional[os.PathLike] = None) -> Tuple[Path, str]:
    """Return ``(root, source)``: argument, else ``MRAC_THESIS_ROOT``, else the importable experiments package."""
    if thesis_root:
        raw, source = Path(thesis_root), "argument"
    elif os.environ.get(ENV_VAR):
        raw, source = Path(os.environ[ENV_VAR]), ENV_VAR
    else:
        raw, source = _installed_root(), "importable experiments package"
        if raw is None:
            raise FileNotFoundError(
                f"No thesis checkout found: the package '{THESIS_PACKAGE}' is not importable. "
                f"Install the thesis checkout editable in this environment (pip install -e <thesis checkout>), "
                f"set {ENV_VAR}, or pass thesis_root."
            )
    root = raw.expanduser().resolve()
    if not (root / "experiments" / "mrac").is_dir():
        raise FileNotFoundError(
            f"The thesis checkout found via {source} has no experiments/mrac; "
            f"set {ENV_VAR} or pass thesis_root."
        )
    return root, source


def resolve_root(thesis_root: Optional[os.PathLike] = None) -> Path:
    """Argument, else ``MRAC_THESIS_ROOT``, else the checkout that provides the importable experiments package."""
    return find_root(thesis_root)[0]


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
    root, source = find_root(thesis_root)
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    importlib.invalidate_caches()

    origins: dict[str, str] = {}
    for name in modules:
        mod = importlib.import_module(name)
        file = Path(mod.__file__ or "").resolve()
        if root not in file.parents:
            raise ImportError(
                f"{name} was imported from outside the thesis checkout found via {source}. "
                "Restart the kernel (another checkout was imported first)."
            )
        origins[name] = file.relative_to(root).as_posix()

    commit = _git(root, "rev-parse", "HEAD") or "UNKNOWN"
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "UNKNOWN"
    status = _git(root, "status", "--porcelain", "--", *DIRTY_SCOPE)
    # porcelain lines are "XY path"; _git strips the leading space of the first line
    dirty_paths = tuple(line.split(None, 1)[1] for line in status.splitlines() if line.strip())
    prov = Provenance(
        root, commit, branch, bool(dirty_paths), sys.version.split()[0], origins, dirty_paths, source
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
        f"Thesis checkout: found via {prov.source}",
        f"Commit: {prov.commit} (branch {prov.branch}"
        + (", UNCOMMITTED CHANGES in " + ", ".join(prov.dirty_paths) if prov.dirty else ", clean for " + ", ".join(DIRTY_SCOPE))
        + ")",
        f"Python: {prov.python}",
        f"Claim limits: {CLAIM_LIMITS}",
    ]
    for name, path in prov.origins.items():
        lines.append(f"import origin: {name} -> <thesis>/{path}")
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
