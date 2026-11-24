"""Presentation layer for the MRAC-RL logbook notebooks (05, 06, 07).

``mrac_lab`` holds no physics and no control law. It only (a) points the
notebook at the thesis repository that owns the tested offline code
(``provenance``), (b) fixes colours and line styles (``style``), (c) draws
standard panels from plain arrays (``plots``), (d) builds the parameter
panels (``widgets``) and (e) runs the experiment grid (``grid``, orchestration only).

Everything shown through it is EXPLORATORY, offline and PROXY (development PC,
reduced taut point-load model). It makes no stability claim and no claim about
the Pymunk plant.
"""

from . import grid, plots, provenance, style, widgets  # noqa: F401

__all__ = ["grid", "plots", "provenance", "style", "widgets"]
