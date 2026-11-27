"""Cached results: locate local run folders safely and never overwrite a cache.

The run folders ``results/mrac_*`` are local and not in Git (``.gitignore``), so a
fresh clone does not have them; ``RESULTS_REGEN.md`` lists how each was produced.
The three grid CSVs directly under ``results/`` are tracked historical evidence.

* ``latest_completed`` picks the newest run folder of a stage whose
  ``receipt.json`` says ``completed`` (an aborted or partial run is never picked).
* ``require`` checks that named run folders exist.
* ``write_new_text`` writes a file only if it does not exist yet.

Paths are relative to this repository (``RESULTS``); nothing here runs episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

#: ``results/`` of this repository, independent of the kernel's working directory.
RESULTS = Path(__file__).resolve().parents[1] / "results"

HINT = (
    "The results/mrac_* run folders are local and not in Git, so a fresh clone does not have "
    "them; RESULTS_REGEN.md lists the command that produced each folder."
)


def status(folder: Path) -> Optional[str]:
    """``receipt.json`` status of a run folder, or None without a readable receipt."""
    receipt = Path(folder) / "receipt.json"
    try:
        return json.loads(receipt.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return None


def latest_completed(prefix: str, results: Path = RESULTS, prefer_composite: bool = True) -> Path:
    """Newest completed run folder ``<prefix>_<timestamp>``.

    With ``prefer_composite``, a completed ``<prefix>_composite_<timestamp>`` folder
    (completion parts combined) is preferred. Only folders whose receipt status is
    ``completed`` qualify; the name order breaks ties between completed runs only.
    """
    results = Path(results)

    def completed(pattern: str) -> list:
        return [p for p in sorted(results.glob(pattern)) if p.is_dir() and status(p) == "completed"]

    runs = (completed(f"{prefix}_composite_*") if prefer_composite else []) or completed(f"{prefix}_2*")
    if not runs:
        raise FileNotFoundError(f"No completed run folder results/{prefix}_* found. {HINT}")
    return runs[-1]


def require(*names: str, results: Path = RESULTS) -> None:
    """Raise with the regeneration hint if any named run folder is missing."""
    missing = [n for n in names if not (Path(results) / n).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing local run folder(s): {', '.join(missing)}. {HINT}")


def write_new_text(path, text: str) -> None:
    """Write ``text`` to a new file; refuse (FileExistsError) if ``path`` already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:  # same newline handling as the former Path.write_text
        fh.write(text)
