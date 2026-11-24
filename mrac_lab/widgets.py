"""Standard parameter panel: ipywidgets controls plus one ``Output`` that is redrawn.

Same pattern as tutorial notebooks 01-02 (``Output`` + redraw), with inline
matplotlib (no ipympl). The redraw function builds a figure, shows it, and the
panel closes it, so repeated redraws do not leak figures.

Usage::

    panel = widgets.param_panel(
        {"m_L": ("slider", 1.0, 8.0, 0.5, 4.0, "m_L [kg]"),
         "mode": ("dropdown", ["a", "b"], "a", "mode")},
        draw,                      # draw(**values) -> matplotlib Figure (or None)
    )

The initial draw runs immediately and re-raises its exception, so a broken panel
fails ``nbconvert --execute`` instead of silently showing nothing. Errors raised
by later slider moves are printed into the panel.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Mapping

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

_STYLE = {"description_width": "110px"}
_LAYOUT = w.Layout(width="360px")


def make_control(spec: tuple) -> w.Widget:
    """Build one widget from ``("slider", lo, hi, step, value, label)`` or
    ``("dropdown", options, value, label)`` or ``("check", value, label)``."""
    kind = spec[0]
    if kind == "slider":
        _, lo, hi, step, value, label = spec
        return w.FloatSlider(
            value=value, min=lo, max=hi, step=step, description=label,
            continuous_update=False, readout_format=".3g",
            style=_STYLE, layout=_LAYOUT,
        )
    if kind == "dropdown":
        _, options, value, label = spec
        return w.Dropdown(
            options=options, value=value, description=label,
            style=_STYLE, layout=_LAYOUT,
        )
    if kind == "check":
        _, value, label = spec
        return w.Checkbox(value=value, description=label, indent=False, layout=_LAYOUT)
    raise ValueError(f"unknown control kind {kind!r}")


class ParamPanel:
    """Controls + output + redraw. ``values`` returns the current parameters."""

    def __init__(self, controls: Mapping[str, tuple], draw: Callable[..., Any]):
        self.draw = draw
        self.controls: Dict[str, w.Widget] = {k: make_control(s) for k, s in controls.items()}
        self.out = w.Output()
        self.last_error: str | None = None
        half = (len(self.controls) + 1) // 2
        items = list(self.controls.values())
        self.box = w.VBox(
            [w.HBox([w.VBox(items[:half]), w.VBox(items[half:])]), self.out]
        )
        for c in self.controls.values():
            c.observe(self._on_change, names="value")

    @property
    def values(self) -> Dict[str, Any]:
        return {k: c.value for k, c in self.controls.items()}

    def _render(self, raise_errors: bool) -> None:
        err: BaseException | None = None
        with self.out:
            clear_output(wait=True)
            try:
                fig = self.draw(**self.values)
                if fig is not None:
                    plt.show()
                    plt.close(fig)
                plt.close("all")
                self.last_error = None
            except Exception as exc:  # shown in the panel, optionally re-raised
                self.last_error = traceback.format_exc()
                print(self.last_error)
                err = exc
        if err is not None and raise_errors:
            raise err

    def _on_change(self, _change: Any) -> None:
        self._render(raise_errors=False)

    def show(self) -> "ParamPanel":
        display(self.box)
        self._render(raise_errors=True)
        return self


def param_panel(controls: Mapping[str, tuple], draw: Callable[..., Any]) -> ParamPanel:
    """Create, display and draw a panel (see module docstring)."""
    return ParamPanel(controls, draw).show()
