"""Synthetic checks of the cached-results helpers and the path-free provenance (Batch 7). No episodes."""

import json
from types import SimpleNamespace

import pytest

from mrac_lab import provenance, results


def _run(root, name, status=None):
    folder = root / name
    folder.mkdir()
    if status is not None:
        (folder / "receipt.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    return folder


def test_latest_completed_skips_aborted_and_partial_runs(tmp_path):
    good = _run(tmp_path, "mrac_ext_ab_20261125T140000", "completed")
    _run(tmp_path, "mrac_ext_ab_20261125T150000", "aborted: preflight failed")  # newer, not completed
    _run(tmp_path, "mrac_ext_ab_20261125T160000")  # newest, no receipt
    _run(tmp_path, "mrac_ext_abc_20261125T170000", "completed")  # other stage, must not match "ab"
    assert results.latest_completed("mrac_ext_ab", results=tmp_path) == good


def test_latest_completed_prefers_a_completed_composite(tmp_path):
    _run(tmp_path, "mrac_ext_abc_20261125T150000", "completed")
    comp = _run(tmp_path, "mrac_ext_abc_composite_20261125T160000", "completed")
    assert results.latest_completed("mrac_ext_abc", results=tmp_path) == comp
    assert results.latest_completed("mrac_ext_abc", results=tmp_path, prefer_composite=False).name.endswith("T150000")


def test_missing_runs_name_the_regeneration_guide(tmp_path):
    _run(tmp_path, "mrac_vanilla_validation_20261125T020307", "disk_below_5GB")
    with pytest.raises(FileNotFoundError, match="RESULTS_REGEN.md"):
        results.latest_completed("mrac_vanilla_validation", results=tmp_path)
    with pytest.raises(FileNotFoundError, match="mrac_long_horizon_x.*RESULTS_REGEN.md"):
        results.require("mrac_long_horizon_x", results=tmp_path)
    results.require(results=tmp_path)  # nothing required: no error


def test_write_new_text_refuses_an_existing_file(tmp_path):
    target = tmp_path / "sub" / "grid.csv"
    results.write_new_text(target, "a,b\n1,2\n")
    with pytest.raises(FileExistsError):
        results.write_new_text(target, "overwritten\n")
    assert target.read_text(encoding="utf-8") == "a,b\n1,2\n"


def _fake_checkout(tmp_path):
    root = tmp_path / "some" / "checkout"
    (root / "experiments" / "mrac").mkdir(parents=True)
    (root / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_root_comes_from_argument_env_or_installed_package(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    monkeypatch.delenv(provenance.ENV_VAR, raising=False)
    assert provenance.find_root(root) == (root.resolve(), "argument")
    monkeypatch.setenv(provenance.ENV_VAR, str(root))
    assert provenance.find_root() == (root.resolve(), provenance.ENV_VAR)
    monkeypatch.delenv(provenance.ENV_VAR)
    fake_spec = SimpleNamespace(origin=str(root / "experiments" / "__init__.py"), submodule_search_locations=None)
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda name: fake_spec)
    assert provenance.find_root() == (root.resolve(), "importable experiments package")
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(FileNotFoundError, match="pip install -e"):
        provenance.find_root()
    assert not hasattr(provenance, "DEFAULT_THESIS_ROOT")


def test_banner_prints_no_absolute_path(tmp_path, capsys):
    root = _fake_checkout(tmp_path)
    prov = provenance.Provenance(root, "0" * 40, "Claude", False, "3.12", {"experiments.mrac.quad": "experiments/mrac/quad.py"},
                                 (), "importable experiments package")
    provenance.banner(prov, "title")
    out = capsys.readouterr().out
    assert str(root) not in out and str(tmp_path) not in out
    assert "Thesis checkout: found via importable experiments package" in out
    assert "experiments.mrac.quad -> <thesis>/experiments/mrac/quad.py" in out


def test_argument_wins_over_env_and_a_checkout_needs_experiments_mrac(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(provenance.ENV_VAR, str(other))
    assert provenance.find_root(root) == (root.resolve(), "argument")
    with pytest.raises(FileNotFoundError, match="has no experiments/mrac"):
        provenance.find_root()  # the env checkout lacks experiments/mrac


def test_setup_records_origins_relative_and_refuses_outside_imports(tmp_path, monkeypatch):
    import sys

    root = _fake_checkout(tmp_path)
    pkg = root / "b7_fake_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        prov = provenance.setup(root, modules=("b7_fake_pkg.mod",))
        assert prov.source == "argument"
        assert prov.origins == {"b7_fake_pkg.mod": "b7_fake_pkg/mod.py"}
        with pytest.raises(ImportError, match="outside the thesis checkout"):
            provenance.setup(root, modules=("json",))  # the standard library is not under the checkout
    finally:  # leave no fake package importable for later tests
        for name in [m for m in sys.modules if m == "b7_fake_pkg" or m.startswith("b7_fake_pkg.")]:
            del sys.modules[name]
        sys.path_importer_cache.pop(str(root), None)
    assert not any(m.startswith("b7_fake_pkg") for m in sys.modules)
