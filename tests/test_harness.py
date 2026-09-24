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
    # parent without fits cannot seed
    p.derive("Model1", "Unfit", [], overwrite=True)
    _raises(ProjectError, p.derive, "Model1", "SeedC", [], seed_from="Unfit")


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
