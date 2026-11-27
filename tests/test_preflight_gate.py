"""Synthetic checks of the preflight gate (protocol 08 AM5). No thesis import, no episodes."""

import math

import pytest

from mrac_lab.preflight_gate import NOT_FIXED_POINT, classify

EXT = dict(rate_limit=-0.05, inclusive=True)  # extensions_runs (AM1)
VAN = dict(rate_limit=0.0, inclusive=False)  # vanilla_runs


def _row(rate=-0.1, inv=5e-9, drift=5e-17):
    return {"rate_per_s": rate, "invariance": inv, "drift": drift}


def test_pass():
    assert classify(_row(), **EXT) == {"status": "pass", "reasons": []}
    assert classify(_row(rate=-1e-3), **VAN)["status"] == "pass"


def test_rate_limit_is_inclusive_for_ext_and_strict_for_vanilla():
    assert classify(_row(rate=-0.05), **EXT)["status"] == "pass"
    assert classify(_row(rate=-0.049), **EXT)["status"] == "fail"
    assert classify(_row(rate=0.0), **VAN)["status"] == "fail"
    # the old missing-direction artefact (~+3e-11 /s) fails both
    assert classify(_row(rate=2.95e-11), **VAN)["status"] == "fail"
    assert classify(_row(rate=2.95e-11), **EXT)["status"] == "fail"


@pytest.mark.parametrize("rate", [math.nan, math.inf, -math.inf])
def test_non_finite_rate_fails(rate):
    r = classify(_row(rate=rate), **EXT)
    assert r["status"] == "fail" and "rate not finite" in r["reasons"]


def test_invariance_is_strict_for_every_variant():
    # the saved abc value (collinear split put in S although sigma_W != sigma) fails
    assert classify(_row(inv=2.63e-5), **EXT)["status"] == "fail"
    assert classify(_row(inv=1e-6), **EXT)["status"] == "fail"
    assert classify(_row(inv=9e-7), **EXT)["status"] == "pass"
    assert classify(_row(inv=math.nan), **EXT)["status"] == "fail"


def test_drift_is_labelled_not_certified_not_failed():
    r = classify(_row(drift=0.0177), **EXT)  # saved veh_delta=0.10 value
    assert r["status"] == "not_certified"
    assert len(r["reasons"]) == 1 and NOT_FIXED_POINT in r["reasons"][0]
    assert classify(_row(drift=1e-9), **EXT)["status"] == "pass"


def test_drift_does_not_mask_a_failure():
    r = classify(_row(rate=0.02, drift=0.0177), **EXT)
    assert r["status"] == "fail"
    assert any("rate" in s for s in r["reasons"]) and any(NOT_FIXED_POINT in s for s in r["reasons"])
    assert classify(_row(drift=math.nan), **EXT)["status"] == "fail"


def test_plant_without_exact_ideal_is_never_gated():
    # vanilla l != l_r: reported, never excludes a design (as before AM5)
    r = classify(_row(rate=0.027), **VAN, ideal_equilibrium=False)
    assert r["status"] == "not_certified"
    assert r["reasons"][0].startswith("no exact ideal") and any("rate" in s for s in r["reasons"])
    assert classify(_row(), **VAN, ideal_equilibrium=False)["status"] == "not_certified"
