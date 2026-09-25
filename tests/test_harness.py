"""
Tests for rsoxr_harness.  Plain asserts; run either way:

    python tests/test_harness.py            (refnxlocal env, no pytest needed)
    pytest tests/test_harness.py

Real JAXNS sampling on CPU is very slow (~100 likelihood evals/s), so
test_jaxns_job_cache_and_skip only runs with RSOXR_SLOW=1; JAXNS itself is
validated on GPU against notebook caches (milestone M4 gate).
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from rsoxr_harness import (FitProject, ProjectError, LayerSpec,  # noqa: E402
                           ModelRecipe, pm_offset, free_parameter_table)
from rsoxr_harness import recipe as R  # noqa: E402
from rsoxr_harness.materials import MaterialSource, resolve, load_table  # noqa: E402
from rsoxr_harness.testing import make_synthetic_project, base_recipe  # noqa: E402


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


_TMP = {}


def synthetic():
    """One synthetic project shared by the tests in this run."""
    if "proj" not in _TMP:
        d = tempfile.mkdtemp(prefix="rsoxr_harness_test_")
        _TMP["dir"] = d
        _TMP["proj"] = make_synthetic_project(d)
    return FitProject.open(_TMP["proj"].path)


# ---------------------------------------------------------------------------
# M1: recipes
# ---------------------------------------------------------------------------

def test_recipe_json_roundtrip():
    r = base_recipe()
    d = json.loads(json.dumps(r.to_json()))
    r2 = ModelRecipe.from_json(d)
    assert r2.to_json() == r.to_json()
    assert r2.physics_hash() == r.physics_hash()
    assert isinstance(r2.layer("Film").thickness_bounds, tuple)


def test_add_layer_above_below_like():
    r = base_recipe()
    r2 = R.add_layer(r, "Film2", above="Film", like="Film", thickness=20,
                     thickness_bounds=(5, 40, True), material="Film")
    assert r2.layer_order == ["air", "Film2", "Film", "SiO2", "Si"]
    assert r2.layer("Film2").roughness == r.layer("Film").roughness   # inherited
    assert r2.layer("Film2").thickness == 20
    assert r.layer_order == ["air", "Film", "SiO2", "Si"]              # untouched
    r3 = R.add_layer(r, "Inter", below="Film", material="Film", thickness=10)
    assert r3.layer_order == ["air", "Film", "Inter", "SiO2", "Si"]
    _raises(ValueError, R.add_layer, r, "Film", above="SiO2")
    _raises(KeyError, R.add_layer, r, "X", above="Nope")
    _raises(ValueError, R.add_layer, r, "X", above="Film", below="Film")
    _raises(ValueError, R.add_layer, r, "X", above="Film", thicknes=3)


def test_other_ops_and_derive():
    r = base_recipe()
    r2 = R.derive(r, "Model2", [
        {"op": "add_layer", "name": "Film2", "above": "Film", "like": "Film",
         "thickness": 20, "thickness_bounds": [5, 40, True]},
        {"op": "set_layer", "layer": "Film", "sld_offset": {"real": [-1, 1, True],
                                                            "imag": [-1, 1, True]}},
        {"op": "set_instrument", "vary_bkg": False},
    ], description="two films")
    assert r2.parent == "Model1" and r2.name == "Model2" and len(r2.ops) == 3
    assert r2.layer("Film").sld_offset["real"] == (-1.0, 1.0, True)
    assert r2.layer("Film2").sld_offset["real"] == (-2.0, 2.0, True)
    assert r2.instrument.vary_bkg is False
    assert R.remove_layer(r2, "Film2").layer_order == r.layer_order
    m = R.move_layer(r2, "Film2", below="Film")
    assert m.layer_order == ["air", "Film", "Film2", "SiO2", "Si"]
    rn = R.rename_layer(r, "Film", "PTD")
    assert "PTD" in rn.layer_order and rn.layer("PTD").material == "Film"
    sm = R.set_material(r, "Film", "SiO2")
    assert sm.layer("Film").material == "SiO2"
    cl = R.set_layer(r, "SiO2", thickness_bounds=None)
    assert cl.layer("SiO2").thickness_bounds is None
    _raises(ValueError, R.apply_op, r, {"op": "explode"})


def test_diff_text():
    r = base_recipe()
    r2 = R.derive(r, "Model2", [{"op": "add_layer", "name": "Film2",
                                 "above": "Film", "like": "Film"}])
    d = "\n".join(R.diff(r, r2))
    assert "+ Film2 (above Film)" in d and "layer order" in d
    r3 = R.set_layer(r, "Film", roughness_bounds=(1, 20, True))
    assert any("Film.roughness_bounds" in s for s in R.diff(r, r3))
    assert R.diff(r, r) == ["(no differences)"]
    assert "Film" in R.layer_table(r2)


def test_validation():
    r = base_recipe()
    bad = R.set_layer(r, "Film", thickness=500)
    errs, _ = R.validate(bad)
    assert any("outside" in e for e in errs)
    bad = R.set_layer(r, "Film", sld_offset={"real": (2, 2, True)})
    errs, _ = R.validate(bad)
    assert any("hi > lo" in e for e in errs)
    errs, _ = R.validate(r, known_materials={"Film", "Si"})
    assert any("SiO2" in e for e in errs)
    errs, warns = R.validate(R.move_layer(r, "air", below="Si"))
    assert not errs and any("top layer" in w for w in warns)


# ---------------------------------------------------------------------------
# M1: materials
# ---------------------------------------------------------------------------

def test_material_resolve_file_and_formula():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    si = resolve(MaterialSource("file", path=os.path.join(here, "OC/Si_SLD.txt")),
                 [285.0])
    assert si.shape[1] == 3 and si[0, 0] < 285 < si[-1, 0]
    f = resolve(MaterialSource("formula", formula="SiO2", density=2.2),
                [280.0, 290.0])
    assert f.shape[1] == 3 and f[0, 0] < 280 and f[-1, 0] > 290
    assert np.all(np.isfinite(f)) and f[:, 1].max() > 0
    _raises(ValueError, MaterialSource, "formula", formula="SiO2")
    _raises(ValueError, MaterialSource, "nonsense", path="x")


def test_load_table_csv_header():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "t.csv")
        with open(p, "w") as fh:
            fh.write("Energy_eV,Real_SLD,Imag_SLD\n290,1,2\n280,3,4\n")
        a = load_table(p)
        assert a.shape == (2, 3) and a[0, 0] == 280       # sorted
    finally:
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# M1: project + build
# ---------------------------------------------------------------------------

def test_project_models_and_build():
    p = synthetic()
    assert p.energies == [280.0, 285.0, 290.0]
    if "Model1" not in p.model_names:
        p.add_model(base_recipe())
    _raises(ProjectError, p.add_model, base_recipe())               # exists
    r2 = p.derive("Model1", "Model2", [
        {"op": "add_layer", "name": "Film2", "above": "Film", "like": "Film",
         "thickness": 20, "thickness_bounds": [5, 40, True]}],
        description="film split", overwrite=True)
    assert p.lineage("Model2") == ["Model1", "Model2"]
    p2 = FitProject.open(p.path)                                     # persisted
    assert p2.get_model("Model2").to_json() == r2.to_json()
    assert "Model2" in p2.models_table()
    assert "+ Film2" in p2.show_model("Model2")

    b1 = p2.build_objectives("Model1")
    b2 = p2.build_objectives("Model2", energies=[285])
    assert sorted(b1["objectives"]) == [280.0, 285.0, 290.0]
    assert list(b2["objectives"]) == [285.0]
    n1 = len(b1["objectives"][285.0].varying_parameters())
    n2 = len(b2["objectives"][285.0].varying_parameters())
    # Film2 adds thick, rough, sld, isld; Model1: Film 4 + SiO2 2 + scale + bkg
    assert n1 == 8 and n2 == 12, (n1, n2)

    # SOG2-style alias: Film2 seeded from the Film table at each energy
    names = {row[0]: row for row in free_parameter_table(b2["objectives"][285.0])}
    arr = p2.material_array("Film")
    re285 = np.interp(285.0, arr[:, 0], arr[:, 1])
    for lay in ("Film", "Film2"):
        _, v, lo, hi = names[f"{lay} - sld"]
        assert abs(v - re285) < 1e-9 and abs(lo - (re285 - 2)) < 1e-9, (lay, v, lo)
    assert "Film2 - thick" in names and names["Film2 - thick"][2:] == (5.0, 40.0)


def test_build_bad_energy_and_missing_material():
    p = synthetic()
    if "Model1" not in p.model_names:
        p.add_model(base_recipe())
    _raises(ProjectError, p.build_objectives, "Model1", energies=[300])
    bad = R.set_material(p.get_model("Model1"), "Film", "Unobtainium")
    bad.name = "Bad"
    _raises(ProjectError, p.add_model, bad)


def test_energy_overrides():
    r = base_recipe()
    r2 = R.set_energy_override(r, "Film", energy=285, sld_real=5.0, sld_imag=0.3)
    r2 = R.set_energy_override(r2, "Film", energy=285, thickness=250)      # merges
    assert len(r2.overrides) == 1 and r2.overrides[0]["thickness"] == 250
    assert r2.physics_hash() != r.physics_hash()
    assert ModelRecipe.from_json(json.loads(json.dumps(r2.to_json()))).overrides == r2.overrides
    assert "override @ 285 eV" in R.layer_table(r2)
    assert any(s.startswith("+ override") for s in R.diff(r, r2))
    assert R.rename_layer(r2, "Film", "PTD").overrides[0]["layer"] == "PTD"
    assert R.remove_layer(r2, "Film").overrides == []
    assert R.clear_energy_override(r2, "Film").overrides == []
    errs, _ = R.validate(R.set_energy_override(r, "Film", energy=285, thickness=999))
    assert any("outside" in e for e in errs)
    _raises(ValueError, R.set_energy_override, r, "Film", energy=285, colour=1)

    p = synthetic()
    if "OvrModel" not in p.model_names:
        m = R.set_energy_override(base_recipe("OvrModel"), "Film", energy=285,
                                  sld_real=5.0, sld_imag=0.3)
        p.add_model(m)
    objs = p.build_objectives("OvrModel")["objectives"]
    t = {n: (v, lo, hi) for n, v, lo, hi in free_parameter_table(objs[285.0])}
    assert t["Film - sld"] == (5.0, 3.0, 7.0), t["Film - sld"]          # ±2 re-centred
    assert t["Film - isld"] == (0.3, 0.0, 2.3), t["Film - isld"]        # lo clamped at 0
    t280 = {n: v for n, v, *_ in free_parameter_table(objs[280.0])}
    assert abs(t280["Film - sld"] - 5.0) > 1                             # other energies untouched
    bad = R.set_energy_override(base_recipe("OvrBad"), "Film", energy=300, sld_real=1)
    e = _raises(ProjectError, p.add_model, bad)
    assert "no data at that energy" in str(e)
    r3, changes = p.edit_model("OvrModel", [{"op": "clear_energy_override"}])
    assert r3.overrides == [] and any("- override" in c for c in changes)


def test_cli_m1():
    from rsoxr_harness.__main__ import main
    p = synthetic()
    if "Model1" not in p.model_names:
        p.add_model(base_recipe())
    assert main(["models", "-p", p.path]) == 0
    assert main(["show", "-p", p.path, "Model1"]) == 0
    assert main(["build", "-p", p.path, "Model1", "--check", "--energies", "285"]) == 0
    assert main(["build", "-p", p.path, "Model1", "--energies", "300"]) == 2


# ---------------------------------------------------------------------------
# M2: jobs + fitting (CPU, tiny settings)
# ---------------------------------------------------------------------------

_TINY = ["--cpu", "--popsize", "16", "--gens", "40", "--tol", "0"]


def _m1(p):
    if "Model1" not in p.model_names:
        p.add_model(base_recipe())


def test_fit_job_foreground():
    import h5py
    from rsoxr_harness import jobs as J
    from rsoxr_harness.__main__ import main
    p = synthetic()
    _m1(p)
    rc = main(["run", "fit", "-p", p.path, "--models", "Model1"] + _TINY)
    assert rc == 0
    job = [j for j in J.list_jobs(p) if j["models"] == ["Model1"]][-1]
    assert job["status"] == "done", job
    out = job["outputs"]["Model1"]
    for e in ("280.0", "285.0", "290.0"):
        assert out["chi2"][e] < out["chi2_initial"][e], out
    assert p.has_fits("Model1")
    with h5py.File(p.h5_path, "r") as f:
        run = f[f"Synth/285.0/Model1/run_{out['runs']['285.0']}"]
        assert run.attrs["recipe_hash"] == p.get_model("Model1").physics_hash()
        assert run.attrs["job_id"] == job["id"]
        assert np.all(np.isfinite(run["parameters/initial_values"][()]))
        assert np.isfinite(run.attrs["chi_sq_initial"])
    # frozen now
    e = _raises(ProjectError, p.add_model, base_recipe(), overwrite=True)
    assert "frozen" in str(e)
    from rsoxr_harness.analysis import chi2_table
    t = chi2_table(p.h5_path, "Synth", ["Model1"])
    assert list(t.index) == [280.0, 285.0, 290.0] and t.notna().all().all()


def test_concurrent_jobs_distinct_runs():
    import h5py
    from rsoxr_harness import jobs as J
    p = synthetic()
    _m1(p)
    params = dict(p.settings("fit"), popsize=8, n_generations=20, tol=0)
    ja = J.new_job(p, "fit", ["Model1"], [285.0], params, "cpu")
    jb = J.new_job(p, "fit", ["Model1"], [285.0], params, "cpu")
    pa, pb = J.launch(p, ja), J.launch(p, jb)
    assert pa.wait(timeout=600) == 0 and pb.wait(timeout=600) == 0
    ra = J.read_job(p, ja["id"])["outputs"]["Model1"]["runs"]["285.0"]
    rb = J.read_job(p, jb["id"])["outputs"]["Model1"]["runs"]["285.0"]
    assert ra != rb, (ra, rb)
    with h5py.File(p.h5_path, "r") as f:
        g = f["Synth/285.0/Model1"]
        ids = {g[f"run_{r}"].attrs["job_id"] for r in (ra, rb)}
    assert ids == {ja["id"], jb["id"]}


def test_failing_job_records_traceback():
    from rsoxr_harness import jobs as J
    p = synthetic()
    _m1(p)
    tmp = os.path.join(_TMP["dir"], "doomed.csv")
    np.savetxt(tmp, p.material_array("Film"), delimiter=",")
    p.add_material("Doomed", kind="file", path=tmp)
    r = R.set_material(p.get_model("Model1"), "Film", "Doomed")
    r.name, r.parent = "DoomedModel", None
    p.add_model(r, overwrite=True)
    os.remove(tmp)
    job = J.new_job(p, "fit", ["DoomedModel"], [285.0],
                    dict(p.settings("fit"), popsize=8, n_generations=5), "cpu")
    J.launch(p, job, detach=False, echo=False)
    job = J.read_job(p, job["id"])
    assert job["status"] == "failed", job["status"]
    assert "FileNotFoundError" in job["error"]
    assert "FileNotFoundError" in J.tail(job["log"], 50)
    assert "failed" in J.status_table(p)


def test_cancel_and_crash_detection():
    from rsoxr_harness import jobs as J
    p = synthetic()
    _m1(p)
    job = J.new_job(p, "fit", ["Model1"], [285.0],
                    dict(p.settings("fit"), popsize=16, n_generations=100000,
                         tol=0), "cpu")
    proc = J.launch(p, job)
    import time
    for _ in range(60):
        if J.read_job(p, job["id"])["status"] == "running":
            break
        time.sleep(0.5)
    assert J.cancel(p, job["id"])["status"] == "cancelled"
    proc.wait(timeout=60)
    assert J.read_job(p, job["id"])["status"] == "cancelled"
    # a 'running' record whose process is gone → crashed
    ghost = J.new_job(p, "fit", ["Model1"], [285.0], {}, "cpu")
    J.update_job(p, ghost["id"], status="running", pid=proc.pid)
    st = {j["id"]: j["status"] for j in J.list_jobs(p)}
    assert st[ghost["id"]] == "crashed"


# ---------------------------------------------------------------------------
# M3: plots + SLD export
# ---------------------------------------------------------------------------

def _two_fitted_models(p):
    from rsoxr_harness.__main__ import main
    _m1(p)
    if not p.has_fits("Model1"):
        assert main(["run", "fit", "-p", p.path, "--models", "Model1"] + _TINY) == 0
    if "Model2" not in p.model_names:
        p.derive("Model1", "Model2", [
            {"op": "add_layer", "name": "Film2", "above": "Film", "like": "Film",
             "thickness": 20, "thickness_bounds": [5, 40, True]},
            {"op": "set_layer", "layer": "Film", "thickness": 230,
             "thickness_bounds": [200, 260, True]}])
    if not p.has_fits("Model2"):
        assert main(["run", "fit", "-p", p.path, "--models", "Model2"] + _TINY) == 0


def test_seed_from_fit():
    from rsoxr_harness.analysis import param_table
    p = synthetic()
    _two_fitted_models(p)                         # Model1 fitted at 280/285/290
    fit = param_table(p.h5_path, "Synth", ["Model1"])
    best = {(r.energy, r.param): r.value for r in fit.itertuples()}
    # wider thickness bounds, start from Model1's fit
    r = p.derive("Model1", "SeedA", [
        {"op": "set_layer", "layer": "Film", "thickness_bounds": [200, 300, True]},
        {"op": "set_layer", "layer": "SiO2", "roughness": 3, "roughness_bounds": [1, 6, False]}],
        seed_from="Model1", overwrite=True)
    assert r.seed["model"] == "Model1" and sorted(r.seed["runs"]) == ["280.0", "285.0", "290.0"]
    assert "seed start from Model1" in p.show_model("SeedA")
    assert r.physics_hash() != R.seed_from_fit(r, None).physics_hash()
    b = p.build_objectives("SeedA")
    assert b["seed_report"]["seeded"] and not b["seed_report"]["clipped"]
    for e, o in b["objectives"].items():
        vals = {q.name: q for q in o.parameters.flattened()}
        for n in ("Film - thick", "Film - sld", "Film - isld", "SiO2 - thick", "scale", "bkg"):
            assert np.isclose(vals[n].value, best[(e, n)]), (e, n)
        assert vals["SiO2 - rough"].value == 3 and not vals["SiO2 - rough"].vary   # fixed: untouched
        assert o.chisqr() < p.build_objectives("Model1", [e])["objectives"][e].chisqr()
    # tight bounds that exclude the fitted thickness → clipped and reported
    ft = best[(285.0, "Film - thick")]
    lo = round(ft) + 3
    r2 = p.derive("Model1", "SeedB", [
        {"op": "set_layer", "layer": "Film", "thickness": lo + 1,
         "thickness_bounds": [lo, lo + 10, True]},
        {"op": "set_energy_override", "layer": "Film", "energy": 290, "thickness": lo + 5}],
        seed_from="Model1", overwrite=True)
    b2 = p.build_objectives("SeedB")
    cl = b2["seed_report"]["clipped"][285.0]
    assert cl[0][0] == "Film - thick" and np.isclose(
        {q.name: q.value for q in b2["objectives"][285.0].parameters.flattened()}["Film - thick"], lo)
    t290 = {q.name: q.value for q in b2["objectives"][290.0].parameters.flattened()}
    assert t290["Film - thick"] == lo + 5                                       # override wins
    # layer-selective seed: only SiO2 (+ instrument) from Model1; Film from the recipe
    r3 = p.derive("Model1", "SeedL", [
        {"op": "set_layer", "layer": "Film", "thickness": 240, "thickness_bounds": [200, 300, True]}],
        seed_from="Model1", seed_layers=["SiO2"], overwrite=True)
    assert r3.seed["layers"] == ["SiO2"] and r3.seed["instrument"] is True
    assert "fitted values of SiO2 + instrument" in p.show_model("SeedL")
    assert r3.physics_hash() != R.seed_from_fit(r3, "Model1").physics_hash()
    b3 = p.build_objectives("SeedL")
    for e, o in b3["objectives"].items():
        v = {q.name: q.value for q in o.parameters.flattened()}
        assert v["Film - thick"] == 240
        for n in ("SiO2 - thick", "scale", "bkg"):
            assert np.isclose(v[n], best[(e, n)]), (e, n)
    r4 = p.derive("Model1", "SeedL2", [], seed_from="Model1", seed_layers=["SiO2"],
                  seed_instrument=False, overwrite=True)
    b4 = p.build_objectives("SeedL2")
    v = {q.name: q.value for q in b4["objectives"][285.0].parameters.flattened()}
    assert v["scale"] == p.get_model("Model1").instrument.scale
    # derive without seed_from drops the parent's seed
    assert p.derive("SeedL", "NoSeed", [], overwrite=True).seed is None
    _raises(KeyError, R.seed_from_fit, r3, "Model1", layers=["Nope"])
    # parent without fits cannot seed
    p.derive("Model1", "Unfit", [], overwrite=True)
    _raises(ProjectError, p.derive, "Model1", "SeedC", [], seed_from="Unfit")


def test_widen_stuck_sld():
    from rsoxr_harness.analysis import param_table
    p = synthetic()
    _two_fitted_models(p)
    fit = param_table(p.h5_path, "Synth", ["Model1"])
    fit = fit[fit.param.isin(["Film - sld", "Film - isld"])]
    # tol 100% → every varying SLD counts as near its closer bound
    r, rep = p.widen_stuck_sld("Model1", "Wide", "Film", step=1.0, tol_pct=100,
                               overwrite=True)
    assert r.seed["model"] == "Model1" and rep
    assert "SLD bounds tab" in p.show_model("Wide")
    b = p.build_objectives("Wide")
    m1 = p.build_objectives("Model1")
    for (e, part, v, lb, ub, side, nlo, nhi) in rep:
        pname = "Film - sld" if part == "real" else "Film - isld"
        q = {x.name: x for x in b["objectives"][e].parameters.flattened()}[pname]
        q1 = {x.name: x for x in m1["objectives"][e].parameters.flattened()}[pname]
        assert np.isclose(q.value, v), (e, part)                  # seeded, not reset
        if side == "upper":
            assert np.isclose(q.bounds.ub, q1.bounds.ub + 1) and np.isclose(q.bounds.lb, q1.bounds.lb)
        elif side == "lower":
            assert np.isclose(q.bounds.lb, q1.bounds.lb - 1) and np.isclose(q.bounds.ub, q1.bounds.ub)
    from rsoxr_harness.plots import plot_sld_vs_energy
    _png_ok(plot_sld_vs_energy(p, "Film", ["Model1"], show_bounds=True, bounds_from=["Wide"]))
    # chained widening: seeds from a model whose fit sits outside the grandparent's
    # window must not be clipped (overrides are applied before seeding)
    if not p.has_fits("Wide"):
        from rsoxr_harness.__main__ import main
        assert main(["run", "fit", "-p", p.path, "--models", "Wide"] + _TINY) == 0
    r2, _ = p.widen_stuck_sld("Wide", "Wide2", "Film", step=2.0, tol_pct=100, overwrite=True)
    fw = param_table(p.h5_path, "Synth", ["Wide"])
    bw = {(x.energy, x.param): x.value for x in fw.itertuples()}
    b2 = p.build_objectives("Wide2")
    assert not b2["seed_report"]["clipped"]
    for e, o in b2["objectives"].items():
        for q in o.parameters.flattened():
            if q.vary:
                assert np.isclose(q.value, bw[(e, q.name)]), (e, q.name)
    # tol 0 → nothing widened, still seeded
    r0, rep0 = p.widen_stuck_sld("Model1", "Wide0", "Film", tol_pct=0, overwrite=True)
    assert not rep0 and not r0.overrides
    # a layer without sld_offset cannot be widened
    _raises(ProjectError, p.widen_stuck_sld, "Model1", "WideX", "SiO2")


def test_link_recipe_ops():
    r = base_recipe()
    assert not r.is_linked
    g = R.derive(r, "G", [{"op": "link", "layer": "Film", "params": ["roughness", "thickness"]}])
    assert g.is_linked and g.links == {"params": {"Film": ["thickness", "roughness"]},
                                       "energies": None}
    assert ModelRecipe.from_json(json.loads(json.dumps(g.to_json()))).to_json() == g.to_json()
    assert g.physics_hash() != r.physics_hash()
    assert "shared across energies: Film thickness+roughness" in R.layer_table(g)
    assert any("links" in d for d in R.diff(r, g))
    assert R.rename_layer(g, "Film", "PTD").links["params"] == {"PTD": ["thickness", "roughness"]}
    assert R.remove_layer(g, "Film").links is None
    assert R.unlink(g, "Film", "roughness").links["params"] == {"Film": ["thickness"]}
    assert R.unlink(g).links is None
    _raises(ValueError, R.link, r, "Film", ["sld"])
    _raises(KeyError, R.link, r, "Nope")
    bad = R.set_energy_override(g, "Film", energy=285, thickness=240)
    assert any("override sets its start" in e for e in R.validate(bad)[0])
    one = R.link_energies(g, [285])
    assert any("at least 2" in e for e in R.validate(one)[0])
    ge = R.link_energies(g, [290], erange=(279, 286))
    assert ge.links["energies"] == [290.0, {"erange": [279.0, 286.0]}]


def test_global_build_and_fit():
    import h5py
    from rsoxr_harness.__main__ import main
    from rsoxr_harness.analysis import param_table, _pick_run, chi2_table
    p = synthetic()
    _m1(p)
    p.derive("Model1", "G1", [{"op": "link", "layer": "Film"},
                              {"op": "link", "layer": "SiO2", "params": "thickness"},
                              {"op": "link_energies", "erange": [279, 291]}],
             overwrite=True)
    assert p.get_model("G1").links["energies"] == [280.0, 285.0, 290.0]   # resolved on save
    b = p.build_global("G1")
    ck = b["checks"]
    assert ck["n_shared"] == 3 and ck["n_energies"] == 3
    assert ck["n_global"] == ck["n_independent"] - 2 * 3
    thick = b["shared_params"][("Film", "thickness")]
    assert all([c for c in st if c.name == "Film"][0].thick is thick for st in b["structures"])
    # linked models fit their own energy set only
    assert main(["run", "fit", "-p", p.path, "--models", "G1", "--energies", "285"] + _TINY) == 2
    assert main(["run", "fit", "-p", p.path, "--models", "G1"] + _TINY) == 0
    assert main(["run", "fit", "-p", p.path, "--models", "G1", "--seed", "3"] + _TINY) == 0
    pt = param_table(p.h5_path, "Synth", ["G1"], criteria="last")
    for n in ("Film - thick", "Film - rough", "SiO2 - thick"):
        v = pt[pt.param == n].value
        assert len(v) == 3 and np.allclose(v, v.iloc[0]), n          # one value everywhere
    assert pt[pt.param == "Film - sld"].value.nunique() == 3            # SLD per energy
    with h5py.File(p.h5_path, "r") as f:
        picks, totals = set(), set()
        for e in (280.0, 285.0, 290.0):
            mg = f[f"Synth/{e}/G1"]
            assert sorted(mg) == ["run_0", "run_1"]
            r = _pick_run(mg, "best")
            picks.add(r)
            totals.add(float(mg[r].attrs["global_chi2_total"]))
            assert json.loads(mg[r].attrs["global_energies"]) == [280.0, 285.0, 290.0]
        assert len(picks) == 1 and len(totals) == 1                    # same global run
        tot = totals.pop()
        other = "run_1" if picks == {"run_0"} else "run_0"
        assert tot <= float(f[f"Synth/280.0/G1/{other}"].attrs["global_chi2_total"])
    t = chi2_table(p.h5_path, "Synth", ["G1"], reduced=False)
    assert np.isclose(t["G1"].sum(), tot)
    # a later model can start from the global fit
    p.derive("G1", "G1s", [], seed_from="G1", overwrite=True)
    bs = p.build_global("G1s")
    best = param_table(p.h5_path, "Synth", ["G1"])
    assert np.isclose(bs["shared_params"][("Film", "thickness")].value,
                      best[best.param == "Film - thick"].value.iloc[0])
    assert np.isclose(bs["checks"]["chi2_total"], tot, rtol=1e-6)


def _stoich_table(path, formula="C8H8", density=1.05):
    """KK-self-consistent [E, Re, Im] table for a known formula/density."""
    from Model_Setup import generate_sld_array_from_material
    from rsoxr_harness.stoich import kk_sld
    E = np.arange(270.0, 331.0, 1.0)
    base = generate_sld_array_from_material(formula, density, E)
    tab = kk_sld(base, formula, density, [270.0, 330.0])
    tab = tab[np.isin(np.round(tab[:, 0], 6), np.round(E, 6))]
    np.savetxt(path, tab, delimiter=",", header="Energy_eV,Real_SLD,Imag_SLD", comments="")
    return tab


def test_stoich_spec():
    from rsoxr_harness.stoich import StoichSpec
    sp = StoichSpec(name="a", source="x.csv", atoms=["C", "H"], counts=[8, [4, 12, 2]],
                    density_range=(0.8, 1.5))
    assert sp.n_candidates() == 5 and sp.counts == [8, [4, 12, 2]]
    assert StoichSpec.from_json(json.loads(json.dumps(sp.to_json()))).to_json() == sp.to_json()
    fr = StoichSpec(name="f", source="x.csv", mode="fragments", fragments=["C8H8", "CH2"],
                    counts=[1, [0, 3]], density_range=(0.8, 1.5))
    assert fr.n_candidates() == 4
    _raises(ValueError, StoichSpec, name="b", source="x", atoms=["C"], counts=[1, 2])
    _raises(ValueError, StoichSpec, name="b", source="x", atoms=["C"], counts=[[5, 2]])
    _raises(ValueError, StoichSpec, name="b", source="x", mode="fragments", counts=[1])
    _raises(ValueError, StoichSpec, name="b", source="x", atoms=["C"], counts=[1],
            density_range=(2, 1))


def test_stoich_split_energy_mask():
    import kk_stoichiometry_fit as K
    from rsoxr_harness.stoich import StoichSpec, rmse_for
    E = np.array([270, 280, 284, 286, 290, 300, 310, 320, 330.0])
    m = K._energy_mask_bool(E, [(270, 284), (310, 330)])
    assert m.tolist() == [True, True, True, False, False, False, True, True, True]
    assert K._energy_mask_bool(E, (284, 300)).sum() == 4 and K._energy_mask_bool(E, None).all()
    assert K._mask_label([(270, 284), (310, 330)]) == "270–284 ∪ 310–330"
    sp = StoichSpec(name="m", source="x", atoms=["C", "H"], counts=[8, [4, 8]],
                    energy_mask=[[270, 284], [310, 330]])
    assert sp.energy_mask == [[270.0, 284.0], [310.0, 330.0]] and "∪" in sp.describe()
    assert StoichSpec(name="m", source="x", atoms=["C"], counts=[1],
                      energy_mask=(270, 284)).energy_mask == [270.0, 284.0]
    _raises(ValueError, StoichSpec, name="m", source="x", atoms=["C"], counts=[1],
            energy_mask=[[284, 270]])
    # a table that is only KK-consistent outside 284–310: the split mask ignores the bad middle
    d = tempfile.mkdtemp()
    tab = _stoich_table(os.path.join(d, "t.csv"))
    bad = tab.copy()
    mid = (bad[:, 0] > 284) & (bad[:, 0] < 310)
    bad[mid, 1] += 3.0
    full = rmse_for(bad, "C8H8", 1.05, [270, 330])
    split = rmse_for(bad, "C8H8", 1.05, [270, 330], [(270, 284), (310, 330)])
    assert full > 1.0 and split < 0.05, (full, split)


def test_stoich_chi2_intervals():
    from rsoxr_harness.__main__ import main
    from rsoxr_harness import plots as P, stoich as S
    p = synthetic()
    d = tempfile.mkdtemp()
    tab = _stoich_table(os.path.join(d, "t.csv"))
    noisy = tab.copy()
    rng = np.random.default_rng(1)
    sig = 0.05
    noisy[:, 1] += rng.normal(0, sig, size=len(tab))   # real only: imag feeds the KK input
    csv = os.path.join(p.path, "ps_sigma.csv")
    np.savetxt(csv, np.column_stack([noisy, np.full((len(tab), 2), sig)]), delimiter=",",
               header="Energy_eV,Real_SLD,Imag_SLD,Real_std,Imag_std", comments="")
    p.add_stoich(name="ps_sig", source=csv, atoms=["C", "H"], counts=[8, [4, 12]],
                 density_range=(0.8, 1.5), n_workers=4, overwrite=True)
    assert main(["stoich", "run", "ps_sig", "-p", p.path]) == 0
    assert main(["stoich", "chi2", "ps_sig", "-p", p.path, "--workers", "4"]) == 0
    import pandas as pd
    df = pd.read_csv(os.path.join(p.stoich_dir("ps_sig"), "chi2.csv"))
    with open(os.path.join(p.stoich_dir("ps_sig"), "chi2.json")) as fh:
        info = json.load(fh)
    assert info["best_formula"] == "C8H8" and abs(info["best_density"] - 1.05) < 0.01
    assert info["nu"] == info["n_points"] - 2
    assert 0.25 < info["birge_s2"] < 1.0, info      # ≈ ½: only the real half carries noise
    df, info, grid = S.load_chi2(p.stoich_dir("ps_sig"))
    iv, _, els = S.chi2_intervals(df, info, grid=grid)
    lo, best, hi, n = iv[1.0]["density"]
    assert lo < best < hi and n >= 1 and hi - lo < 0.01, iv[1.0]["density"]
    lo4, _, hi4, _ = iv[4.0]["density"]
    assert lo4 <= lo and hi4 >= hi and lo4 <= 1.05 <= hi4, iv[4.0]["density"]
    assert iv[4.0]["w_C"][3] >= iv[1.0]["w_C"][3]
    _png_ok(P.plot_stoich_chi2(p, "ps_sig"))
    _png_ok(P.plot_stoich_chi2_rows([(p, "ps_sig", "A"), (p, "ps_sig", "B")]))
    # a source without σ columns is refused
    _raises(ValueError, S.load_sigma, os.path.join(d, "t.csv"), tab[:, 0])


def test_stoich_cpu_cap():
    from rsoxr_harness import stoich as S
    cap = S.cpu_cap()
    assert cap == max(1, (os.cpu_count() or 2) // 2)
    assert S.resolve_workers(None) == cap and S.resolve_workers(10 ** 6) == cap
    assert S.resolve_workers(3, in_use=cap) == 1 and S.resolve_workers(None, in_use=cap - 2) == 2
    old = S._REGISTRY
    S._REGISTRY = os.path.join(tempfile.mkdtemp(), "reg.json")
    try:
        n = S.reserve_workers(None)
        assert n == cap and S.cpus_in_use() == cap
        S.release_workers()
        assert S.cpus_in_use() == 0
    finally:
        S._REGISTRY = old


def test_stoich_search_job_and_plots():
    from rsoxr_harness.__main__ import main
    from rsoxr_harness import plots as P, stoich as S
    p = synthetic()
    csv = os.path.join(p.path, "ps_true.csv")
    tab = _stoich_table(csv)
    p.add_stoich(name="ps_atoms", source=csv, atoms=["C", "H"], counts=[8, [4, 12, 2]],
                 density_range=(0.8, 1.5), n_workers=4, overwrite=True)
    p.add_stoich(name="ps_frag", source=csv, mode="fragments", fragments=["C8H8", "CH2"],
                 counts=[1, [0, 3]], density_range=(0.8, 1.5), n_workers=4, overwrite=True)
    assert main(["stoich", "run", "ps_atoms", "ps_frag", "-p", p.path]) == 0
    sa = S.load_result(p.stoich_dir("ps_atoms"))[0]
    b = S.final_best(sa)
    assert b["formula"] == "C8H8" and abs(b["density"] - 1.05) < 2e-3 and b["rmse"] < 1e-3, b
    assert sa["best"]["counts"] == {"C": 8, "H": 8}
    sf = S.final_best(S.load_result(p.stoich_dir("ps_frag"))[0])
    assert sf["formula"] in ("C8H8", "H8C8") and abs(sf["density"] - 1.05) < 2e-3, sf
    assert "C8H8" in p.stoich_table()
    assert main(["stoich", "show", "ps_atoms", "-p", p.path]) == 0
    # simulation without fitting reproduces the table
    assert S.rmse_for(tab, "C8H8", 1.05, [270, 330]) < 1e-3
    assert S.rmse_for(tab, "C8H12", 1.05, [270, 330]) > 10 * S.rmse_for(tab, "C8H8", 1.05, [270, 330])
    for path in (P.plot_stoich_results(p, "ps_atoms"),
                 P.plot_stoich_density(p, "ps_atoms"),
                 P.plot_stoich_counts(p, "ps_atoms"),
                 P.plot_stoich_counts(p, "ps_frag"),
                 P.plot_stoich_overlay(p, "ps_atoms", formulas=[("C8H10", 1.0)]),
                 P.plot_stoich_overlay(p, source=csv, formulas=[("C8H8", 1.05)])):
        _png_ok(path)
    assert main(["plot", "stoich-overlay", "-p", p.path, "--name", "ps_atoms",
                 "--sim", "C8H12:1.1"]) == 0
    acc, df, best, els = P.stoich_accepted(p, "ps_atoms", tol_pct=5)
    assert list(acc.formula) == ["C8H8"] and els == ["C", "H"]
    assert np.isclose(acc.w_C[0] + acc.w_H[0], 1.0)
    _png_ok(P.plot_stoich_accept(p, "ps_atoms", tol_pct=500))
    _raises(ProjectError, p.get_stoich, "nope")


def _png_ok(path):
    assert os.path.exists(path) and os.path.getsize(path) > 5000, path
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert os.path.exists(os.path.splitext(path)[0] + ".json")


def test_all_plots_write_png():
    from rsoxr_harness import plots as P
    p = synthetic()
    _two_fitted_models(p)
    ms = ["Model1", "Model2"]
    _png_ok(P.plot_chi2_bar(p, ms))
    _png_ok(P.plot_chi2_bar(p, ms, metric="raw", log=True))
    _png_ok(P.plot_chi2_vs_energy(p, ms))
    _png_ok(P.plot_sld_vs_energy(p, "Film", ms, references=["Film"]))
    _png_ok(P.plot_sld_vs_energy(p, "Film2", ms, components=("imag",)))  # only Model2
    _png_ok(P.plot_parameter(p, "Film - thick", ms))
    _png_ok(P.plot_reflectivity_grid(p, ms, erange=(280, 290), ncols=2))
    _png_ok(P.plot_reflectivity_grid(p, ms, energies=[285], residuals=False, q4=True))
    _png_ok(P.plot_sld_profiles(p, ms, references=["Film"]))
    _raises(KeyError, P.plot_sld_vs_energy, p, "Nope", ms)
    _raises(KeyError, P.plot_chi2_bar, p, ["Model9"])
    _raises(KeyError, P.plot_reflectivity_grid, p, ms, energies=[300])


def test_sld_export_roundtrip_and_register():
    from Model_Setup import load_material_sld_array
    from rsoxr_harness import plots as P
    from rsoxr_harness.analysis import param_table
    p = synthetic()
    _two_fitted_models(p)
    path = P.export_sld(p, "Model2", "Film", register_as="Film_M2fit")
    arr = load_material_sld_array(path, verbose=False)
    t = param_table(p.h5_path, "Synth", ["Model2"])
    want = t[t.param == "Film - sld"].sort_values("energy")["value"].values
    assert arr.shape == (3, 3) and np.allclose(arr[:, 1], want)
    assert FitProject.open(p.path).material("Film_M2fit").kind == "fit_csv"
    # exact-name matching: exporting 'Film' must not pick up 'Film2'
    f2 = P.export_sld(p, "Model2", "Film2")
    assert not np.allclose(load_material_sld_array(f2, verbose=False)[:, 1], arr[:, 1])


def test_cli_plots():
    from rsoxr_harness.__main__ import main
    p = synthetic()
    _two_fitted_models(p)
    base = ["-p", p.path, "--models", "Model1", "Model2"]
    assert main(["plot", "chi2"] + base) == 0
    assert main(["plot", "sld-energy"] + base + ["--layer", "Film", "--ref", "Film"]) == 0
    assert main(["plot", "refl"] + base + ["--energies", "285"]) == 0
    assert main(["plot", "sld-energy"] + base + ["--layer", "Nope"]) == 2
    assert main(["fits", "-p", p.path, "--per-energy"]) == 0


# ---------------------------------------------------------------------------
# M4: uncertainty (CPU, tiny settings)
# ---------------------------------------------------------------------------

_TINY_JAXNS = {"num_live_points": 25, "max_samples": 600, "seed": 0,
               "n_posterior_samples": 200, "normalize": True}
_TINY_NUTS = {"n_chains": 2, "n_warmup": 50, "samples_per_chunk": 50,
              "min_samples": 50, "max_samples": 100, "rhat_threshold": 1.5,
              "ess_min": 5, "normalize": True, "seed": 0}


def _sample(p, kind, model, energies, params, **extra):
    from rsoxr_harness import jobs as J
    job = J.new_job(p, kind, [model], energies, dict(params, **extra), "cpu")
    J.launch(p, job, detach=False, echo=False)
    job = J.read_job(p, job["id"])
    assert job["status"] == "done", J.tail(job["log"], 40)
    return job


def test_load_best_reproduces_stored_chi2():
    import h5py
    from rsoxr_harness.analysis import _pick_run
    p = synthetic()
    _two_fitted_models(p)
    b = p.load_best("Model2")
    assert b["source"] == "recipe" and sorted(b["objectives"]) == [280.0, 285.0, 290.0]
    with h5py.File(p.h5_path, "r") as f:
        for e, o in b["objectives"].items():
            g = f[f"Synth/{e}/Model2"]
            stored = g[_pick_run(g, "best")].attrs["chi_sq_final"]
            assert np.isclose(o.chisqr(), stored, rtol=1e-9)
    ext = p.load_best("Model2", [285], h5=p.h5_path, sample="Synth")
    assert ext["source"] == "h5-reconstructed"
    assert np.isclose(ext["objectives"][285.0].chisqr(),
                      b["objectives"][285.0].chisqr(), rtol=1e-6)
    _raises(ProjectError, p.load_best, "Model2", [300])


SLOW = os.environ.get("RSOXR_SLOW") == "1"


def _fake_jaxns(p, model, energy, log_z):
    """Write a cache entry shaped like the worker's (samples near the best fit)."""
    from types import SimpleNamespace
    from Plotting_Refl import save_jaxns
    from gpu_reflect import extract_free_params
    obj = p.load_best(model, [energy])["objectives"][float(energy)]
    pars = extract_free_params(obj)
    rng = np.random.default_rng(0)
    cols = []
    for q in pars:
        lo, hi = q.bounds.lb, q.bounds.ub
        cols.append(np.clip(q.value + 0.01 * (hi - lo) * rng.standard_normal(200), lo, hi))
    smp = np.column_stack(cols)
    med = np.median(smp, axis=0)
    res = SimpleNamespace(samples=smp, log_L_samples=np.zeros(200), log_Z_mean=log_z,
                          log_Z_std=0.3, ESS=150.0, H_mean=5.0, posterior_mean=smp.mean(0),
                          posterior_std=smp.std(0), posterior_median=med,
                          param_names=[q.name for q in pars], run_seconds=0.0)
    path = p.unc_path(model, "jaxns", energy)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_jaxns(path, res)
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump({"method": "jaxns", "model": model, "energy": float(energy),
                   "params": {"normalize": True}, "log_Z_mean": log_z, "log_Z_std": 0.3,
                   "ESS": 150.0, "H_mean": 5.0, "chi2_best": float(obj.chisqr())}, fh)


def test_jaxns_job_cache_and_skip():
    if not SLOW:
        print("  (skipped: set RSOXR_SLOW=1 to run real JAXNS on CPU)")
        return
    import time
    from rsoxr_harness import analysis as A
    p = synthetic()
    _two_fitted_models(p)
    job = _sample(p, "jaxns", "Model1", [285.0], _TINY_JAXNS)
    meta = A.cached_uncertainty(p, "Model1", "jaxns")[285.0]
    assert np.isfinite(meta["log_Z_mean"]) and meta["params"]["normalize"] is True
    assert meta["recipe_hash"] == p.get_model("Model1").physics_hash()
    assert os.path.exists(p.unc_path("Model1", "jaxns", 285.0))
    t0 = time.time()
    job2 = _sample(p, "jaxns", "Model1", [285.0], _TINY_JAXNS)
    assert job2["outputs"]["skipped_cached"] == [285.0] and not job2["outputs"]["energies"]
    r = A.load_uncertainty(p, "Model1", "jaxns")[285.0]
    assert r.samples.shape[1] == len(meta["param_names"])


def test_nuts_job():
    from rsoxr_harness import analysis as A
    p = synthetic()
    _two_fitted_models(p)
    _sample(p, "nuts", "Model1", [285.0], _TINY_NUTS)
    meta = A.cached_uncertainty(p, "Model1", "nuts")[285.0]
    assert np.isfinite(meta["rhat_max"]) and meta["total_samples"] >= 50
    t = A.posterior_table(p, ["Model1"], "nuts")
    row = t[t.param == "Film - thick"].iloc[0]
    assert row.lo <= row["median"] <= row.hi


def test_uncertainty_plots_and_evidence():
    from rsoxr_harness import plots as P, analysis as A
    p = synthetic()
    _two_fitted_models(p)
    for m, lz in (("Model1", -20.0), ("Model2", -18.5)):
        if 285.0 not in A.cached_uncertainty(p, m, "jaxns"):
            if SLOW:
                _sample(p, "jaxns", m, [285.0], _TINY_JAXNS)
            else:
                _fake_jaxns(p, m, 285.0, lz)
    s = A.sum_log_evidence(p, ["Model1", "Model2"])
    assert list(s["n_energies"]) == [1, 1] and s["delta_vs_best"].max() == 0
    if not SLOW:
        assert s.index[0] == "Model2" and np.isclose(s.loc["Model1", "delta_vs_best"], -1.5)
    _png_ok(P.plot_uncertainty(p, "Model1", "jaxns", n_curves=30))
    _png_ok(P.plot_posterior_vs_energy(p, ["Model1", "Model2"], "jaxns",
                                       layer="Film", references=["Film"]))
    _png_ok(P.plot_posterior_vs_energy(p, ["Model1"], "jaxns", param="Film - thick"))
    _png_ok(P.plot_evidence(p, ["Model1", "Model2"]))
    _png_ok(P.plot_corner(p, "Model1", "jaxns", 285))
    if A.cached_uncertainty(p, "Model1", "nuts"):
        _png_ok(P.plot_uncertainty(p, "Model1", "nuts", n_curves=30))
    _raises(KeyError, P.plot_uncertainty, p, "Model2", "nuts")
    from rsoxr_harness.__main__ import main
    assert main(["unc", "-p", p.path, "--models", "Model1", "Model2"]) == 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [(n, f) for n, f in globals().items()
             if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception:
            failed += 1
            print(f"FAIL  {n}")
            traceback.print_exc()
    if "dir" in _TMP:
        shutil.rmtree(_TMP["dir"], ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
