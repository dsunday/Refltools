"""
Tests for rsoxr_reduce.  Plain asserts; run either way:

    python tests/test_reduce_harness.py      (refnxlocal env, no pytest needed)
    pytest tests/test_reduce_harness.py

Synthetic-beamtime tests always run.  The legacy-equivalence and
manual-recovery tests use the Nov 2025 data and are skipped if absent.
"""

import os
import shutil
import sys
import tempfile
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from rsoxr_reduce import inventory as I  # noqa: E402
from rsoxr_reduce import selection as SEL  # noqa: E402
from rsoxr_reduce import settings as S  # noqa: E402
from rsoxr_reduce import stitch as ST  # noqa: E402
from rsoxr_reduce import recover as REC  # noqa: E402
from rsoxr_reduce.project import ReductionProject, ProjectError  # noqa: E402
from rsoxr_reduce.testing import make_beamtime, true_r  # noqa: E402

NOV = "/homes/dfs1/Reflectivity/Data_Reduction/Nov 2025"
_TMP = {}


def beamtime():
    if "bt" not in _TMP:
        d = tempfile.mkdtemp(prefix="rsoxr_reduce_test_")
        _TMP["dir"] = d
        _TMP["bt"] = make_beamtime(d)
    return _TMP["bt"]


def project():
    if "proj" not in _TMP:
        raw, T = beamtime()
        p = ReductionProject.create(raw, log=T["log"], open_beams=[T["open_beam"]],
                                    sample="S1")
        p.run(write=True, diagnostics="flagged")
        _TMP["proj"] = p
    return _TMP["proj"]


# ---------------------------------------------------------------------------
# R1: inventory + selection
# ---------------------------------------------------------------------------
def test_log_parsing_gain_slit_filter():
    raw, T = beamtime()
    log = I.parse_log(T["log"])
    L = log["S1_C_PD000001.dat"]
    assert L["detector"] == "pd" and L["pd_gain"] == 9.0
    assert abs(L["energy"] - 270.0) < 1e-6 and L["slit_gap"] == T["slit"]
    assert L["filter"] == "3 mm Ti 600 nm"
    c = log["S1_C_CT1000001.dat"]
    assert c["detector"] == "cem" and c["count_time"] == 1.0 and c["pd_gain"] is None


def test_classification():
    raw, T = beamtime()
    st = S.defaults()
    files = I.inventory(raw, I.parse_log(T["log"]), st)
    cls = {f.name: f.cls for f in files}
    assert sum(v == "angle" for v in cls.values()) == 5 + 2 + 5 + 5   # PD + extras + CT1 + CT2
    assert list(cls.values()).count("energy_scan") == 1
    assert list(cls.values()).count("empty") + list(cls.values()).count("dummy") == 1
    obs = I.find_open_beams([T["open_beam"]], I.parse_log(T["log"]), st)
    assert len(obs) == 1 and obs[0].pd_gain == 8.0 and obs[0].edge == "C"


def test_selection_reasons():
    raw, T = beamtime()
    st = S.defaults()
    sel = SEL.select(I.inventory(raw, I.parse_log(T["log"]), st), st)
    assert [s["id"] for s in sel["sections"]] == ["PD", "CT1", "CT2"]
    by_e = {round(r["energy"]): r for r in sel["energies"]}
    assert set(by_e) == {270, 280, 285, 290, 300}
    codes = {x["file"]: x["code"] for r in sel["energies"] for x in r["rejected"]}
    # 285: two full PD scans -> the later one wins
    assert by_e[285]["chosen"]["PD"] == "S1_C_PD000004.dat"
    assert codes["S1_C_PD000003.dat"] == "SUPERSEDED"
    # 290: early 0-4° scan is partial
    assert codes["S1_C_PD000005.dat"] == "PARTIAL_RANGE"
    assert by_e[290]["chosen"]["PD"] == "S1_C_PD000006.dat"


# ---------------------------------------------------------------------------
# R2: trims, background, stitching, QC
# ---------------------------------------------------------------------------
def test_folder_clip_and_background():
    p = project()
    lr = p.cfg["last_run"]
    T = beamtime()[1]
    assert abs(lr["clip_angle"] - T["clip_angle"]) <= 0.081, lr["clip_angle"]
    bg = lr["pd_background"]
    assert bg["source"].startswith("pd-vs-ct"), bg
    assert abs(bg["bg"] - T["pd_bg"]) < 5e-4, bg


def test_reduced_curve_matches_truth():
    p = project()
    for key, v in p.cfg["last_run"]["energies"].items():
        E = float(key)
        if v["status"] not in ("ok", "accepted"):
            continue
        d = np.loadtxt(v["output"], skiprows=1)
        th = REC.q_to_theta(d[:, 0], E)
        # compare where R is well measured (fringe maxima region, low/mid angle)
        m = (th > 1) & (th < 25) & (true_r(th, E) > 1e-4)
        ratio = np.median(d[m, 1] / true_r(th[m], E))
        # absolute scale: gain-corrected open beam -> R should be ~1× truth
        assert abs(ratio - 1) < 0.05, (E, ratio)


def test_pd_floor_points():
    """PD kept past its SNR floor: covered points and isolated dropouts are not
    written, but still set the stitch scale."""
    from rsoxr_reduce import trim as TR
    st = S.defaults()
    th = np.arange(0, 10.01, 0.25)
    pd = np.c_[th, 0.002 + 0.01 * np.exp(-th)]           # bg 0.002, signal decays
    i_drop = int(np.argmin(abs(th - 7.0)))
    pd[i_drop, 1] = 0.002 + 1e-6                          # dropout at the floor
    ct = np.c_[np.arange(8, 15.01, 0.25), np.ones(29)]
    lo = {"id": "PD", "kind": "pd", "data": pd, "bg": 0.002, "sigma": 1e-4}
    hi = {"id": "CT1", "kind": "cem", "data": ct}
    cap, last = int(np.argmin(abs(th - 5.0))), len(th) - 2
    idx, why, fl = TR.pd_floor_points(lo, hi, 0, last, cap, st)
    assert i_drop in idx, (idx, why)
    assert all(th[i] >= 8 for i in idx if i != i_drop), idx      # rest = covered by CT1
    assert not [i for i in range(cap + 1, last + 1) if th[i] < 8 and i != i_drop and i in idx]
    assert fl and fl[0]["code"] == "PD_FLOOR_POINTS"
    # no extension past the floor -> nothing dropped
    assert TR.pd_floor_points(lo, hi, 0, cap, cap, st)[0] == []
    # stitch: dropped points absent from the output, scale unchanged
    a = ST.reduce_sections([pd, ct], [(1, -1), (0, -1)], [0.002, 0.0], 285, i0=1.0)
    b = ST.reduce_sections([pd, ct], [(1, -1), (0, -1)], [0.002, 0.0], 285, i0=1.0,
                           point_drops={0: idx})
    assert a["sections"][1]["scale"] == b["sections"][1]["scale"]
    assert len(b["q_data"]) == len(a["q_data"]) - len(idx)
    assert b["sections"][0]["n_floor_dropped"] == len(idx)


def test_counting_error_model():
    """rel_errs add in quadrature to the flat error; R and Q are unchanged."""
    pd = np.c_[np.arange(0, 5.01, 0.25), np.linspace(1.0, 0.01, 21)]
    rel = np.full(len(pd), 0.2)
    a = ST.reduce_sections([pd], [(1, -1)], [0.0], 285, i0=1.0)
    b = ST.reduce_sections([pd], [(1, -1)], [0.0], 285, i0=1.0, rel_errs={0: rel})
    assert np.allclose(a["q_data"][:, :2], b["q_data"][:, :2])
    assert np.allclose(a["q_data"][:, 2] / a["q_data"][:, 1], 0.01)
    assert np.allclose(b["q_data"][:, 2] / b["q_data"][:, 1], np.sqrt(0.01 ** 2 + 0.2 ** 2))
    assert S.defaults()["output"]["error_model"] == "flat"


def test_gain_correction_and_filter_change_flags():
    p = project()
    plan = p.plan()
    pe = next(x for x in plan["energies"] if abs(x["energy"] - 280) < 0.1)
    assert pe["i0_factor"] == 10.0
    assert any(f["code"] == "GAIN_CORRECTED" for f in pe["flags"])
    v300 = p.cfg["last_run"]["energies"]
    k300 = next(k for k in v300 if abs(float(k) - 300) < 0.1)
    assert "FILTER_CHANGE" in v300[k300]["review"]
    assert v300[k300]["status"] == "review"


def test_legacy_equivalence_synthetic():
    """New stitch == RSoXRProcessor.reduce_data_with_open_beam_option."""
    import contextlib
    import io
    from RSoXRReduction import RSoXRProcessor
    raw, T = beamtime()
    files = [os.path.join(raw, n) for n in
             ("S1_C_PD000002.dat", "S1_C_CT1000002.dat", "S1_C_CT2000002.dat")]
    scans = [np.loadtxt(f, skiprows=1) for f in files]
    trims, bgs = [(5, -30), (0, -1), (2, -3)], [0.003, 0.0, 0.0]
    ob = os.path.join(T["open_beam"], "open_beam_carbon000001.dat")
    new = ST.reduce_sections(scans, trims, bgs, 280.0, i0=ST.OpenBeam(ob)(280.0))["q_data"]
    with contextlib.redirect_stdout(io.StringIO()):
        P = RSoXRProcessor()
        P.load_open_beam_file(ob)
        old = P.reduce_data_with_open_beam_option(scans, list(trims), bgs, 280.0,
                                                  use_open_beam=True)
    assert new.shape == old.shape
    assert np.max(np.abs(new / old - 1)) < 1e-12


# ---------------------------------------------------------------------------
# R4: project, overrides, outputs, CLI
# ---------------------------------------------------------------------------
def test_outputs_written():
    p = project()
    for f in ("reduction.json", "reduction_record.csv", "reduction_report.md",
              "angular_ranges.csv", "overview_scales.png", "overview_waterfall.png"):
        assert os.path.exists(os.path.join(p.path, f)), f
    held = [k for k, v in p.cfg["last_run"]["energies"].items() if v["status"] == "review"]
    for k in held:
        assert not os.path.exists(os.path.join(p.path, p.out_name(float(k))))


def _import_pattern(fn):
    import re
    return re.match(r".*_([0-9.]+).*\.dat", fn)


def test_output_names_match_fitting_import():
    p = project()
    for v in p.cfg["last_run"]["energies"].values():
        if v["output"]:
            assert _import_pattern(os.path.basename(v["output"]))


def test_override_accept_and_refusals():
    p = project()
    k300 = next(k for k in p.cfg["last_run"]["energies"] if abs(float(k) - 300) < 0.1)
    p.override(300, section="PD", start=6, reason="test override")
    p.accept(300, reason="fringes checked")
    plan, out = p.run(write=True, diagnostics="none")
    v = p.cfg["last_run"]["energies"][k300]
    assert v["status"] == "accepted" and v["sections"][0]["start"] == 6
    assert os.path.exists(v["output"])
    p.clear_override(300)
    e = None
    try:
        ReductionProject.create(beamtime()[0], out=os.path.join(_TMP["dir"], "x", "Reduced"))
    except ProjectError as err:
        e = err
    assert e is not None and "Reduced" in str(e)


def test_energy_scans():
    p = project()
    res = p.energy_scans(write=True)
    assert len(res) == 1 and res[0]["data"] is not None
    d = res[0]["data"]
    # (0.05+...-bg)/(i0) with gain correction -> order 0.02
    assert 0.005 < np.median(d[:, 1]) < 0.05
    assert os.path.exists(res[0]["path"])


def test_cli_smoke():
    from rsoxr_reduce.__main__ import main
    p = project()
    for args in (["show"], ["inventory"], ["select"], ["plan"], ["review"], ["why", "285"]):
        main(args + ["-p", p.path])


# ---------------------------------------------------------------------------
# real data (skipped without Nov 2025)
# ---------------------------------------------------------------------------
def test_recover_manual_and_legacy_on_disk():
    b = os.path.join(NOV, "Brewer10", "Carbon")
    if not os.path.isdir(b):
        print("   (skipped: Nov 2025 data not found)")
        return
    files = [os.path.join(b, n) for n in ("BrewerS10_CPD000005.dat", "BrewerS10_CCT1000006.dat",
                                          "BrewerS10_CCT2000006.dat")]
    red = os.path.join(b, "Reduced", "BrewerS_285.0eV.dat")
    rec = REC.recover(red, 285.0008, files)
    assert rec["explained"] == rec["n_red"]
    assert [(h["start"], h["end"]) for h in rec["sections"]] == [(4, -11), (0, -1), (0, -1)]
    assert abs(rec["sections"][0]["bg"] - 0.004) < 1e-9
    ob = os.path.join(NOV, "OpenBeam", "open_beam_carbon_saturday000002.dat")
    scans = [np.loadtxt(f, skiprows=1) for f in files]
    new = ST.reduce_sections(scans, [(4, -11), (0, -1), (0, -1)], [0.004, 0.0, 0.0],
                             285.0008, i0=ST.OpenBeam(ob)(285.0008))["q_data"]
    disk = np.loadtxt(red, skiprows=1)
    assert np.max(np.abs(new / disk - 1)) < 1e-12


# ---------------------------------------------------------------------------
def _run_all():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
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
