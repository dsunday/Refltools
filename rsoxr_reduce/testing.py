"""
testing.py
----------
Synthetic beamtime for tests: a log, raw PD/CEM scans with a known R(θ),
beam clipping below 0.4°, a PD dark offset, a known PD/CEM sensitivity,
Poisson CEM noise, an open beam, and the usual junk (dummy files, a partial
early re-scan, a later duplicate, an energy scan).

    root, truth = make_beamtime(tmpdir)
"""

import os
from datetime import datetime, timedelta

import numpy as np

TRUTH = {"clip_angle": 0.4, "pd_bg": 0.003, "pd_sigma": 0.0004, "k": 8e-7,
         "i0": 2.5, "slit": 169.2, "gain": 9.0}


def true_r(theta, energy):
    """Flat below ~1°, power-law decay, Kiessig fringes above 2° (realistic
    PD ~1e-2 V at 10°, CEM ~1e4 cps at 9° and ~300 cps at 30°)."""
    t = np.asarray(theta, float)
    w = 1.5 + 0.002 * (energy - 270)
    fr = 1 + 0.3 * np.clip((t - 2) / 2, 0, 1) * np.cos(2 * np.pi * t / 1.3)
    return 0.8 * (1 + (t / w) ** 2) ** -1.5 * fr


def _row(name, when, det, gain_count, energy, theta_end, x=-10.0, y=25.0, scan="script",
         filt="3 mm", slit=TRUTH["slit"]):
    r = [""] * 44
    r[0], r[1], r[2] = "2013b0001", "1.00", name
    r[4] = when.strftime("%a, %b %d, %Y")
    r[5] = when.strftime("%I:%M:%S %p")
    r[6] = "photodiode" if det == "pd" else "cem digital"
    r[7] = "analog" if det == "pd" else "digital"
    r[8] = f"{gain_count:g}"
    r[9] = "F"
    r[10] = "100 msec" if det == "pd" else "5"
    r[11], r[16], r[18], r[19] = "m3 mirror", scan, "1200", f"{slit}"
    r[21], r[22], r[24], r[25] = filt, "Ti 600 nm", "150", "stage1"
    r[26], r[27] = f"{energy:.4f}", "eV"
    vals = [theta_end, 0.0, 0.0, 2 * theta_end, 0.0, x, y, -0.5]
    for k, v in enumerate(vals):
        r[28 + 2 * k] = f"{v:.4f}"
        r[29 + 2 * k] = "deg" if k < 4 else "mm"
    return "\t".join(r)


def _write(path, when, cols):
    with open(path, "w") as fh:
        fh.write(when.strftime("%A, %B %d, %Y %I:%M:%S %p") + "\n")
        for row in zip(*cols):
            fh.write("\t".join(f"{v:.6g}" for v in row) + "\n")


def make_beamtime(root, energies=(270.0, 280.0, 285.0, 290.0, 300.0), seed=0):
    rng = np.random.default_rng(seed)
    T = TRUTH
    raw = os.path.join(root, "Sample1", "Carbon")
    obd = os.path.join(root, "OpenBeam")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(obd, exist_ok=True)
    rows = []
    t = datetime(2025, 11, 15, 8, 0, 0)
    q = 3.052e-4

    def pd_scan(name, energy, lo=0.0, hi=12.0, n=151):
        nonlocal t
        th = np.linspace(lo, hi, n)
        clip = np.clip(th / T["clip_angle"], 0, 1)
        v = T["i0"] * true_r(th, energy) * clip + T["pd_bg"]
        v = v + rng.normal(0, T["pd_sigma"], n)
        v = np.round(v / q) * q
        _write(os.path.join(raw, name), t, [th, v, np.full(n, 1e-3), np.full(n, 500.2)])
        rows.append(_row(name, t, "pd", T["gain"], energy, hi))
        t += timedelta(minutes=1)

    def ct_scan(name, energy, lo, hi, n, ct=1.0, filt="3 mm"):
        nonlocal t
        th = np.linspace(lo, hi, n)
        rate = T["i0"] * true_r(th, energy) / T["k"]
        counts = rng.poisson(rate * ct)
        _write(os.path.join(raw, name), t, [th, counts / ct, np.full(n, 1e-3), np.full(n, 500.2)])
        rows.append(_row(name, t, "cem", ct, energy, hi, filt=filt))
        t += timedelta(minutes=2)

    k = 0
    for E in energies:
        k += 1
        if E == 290.0:          # failed early partial scan, then the real one
            pd_scan(f"S1_C_PD{k:06d}.dat", E, 0.0, 4.0, 51)
            k += 1
        pd_scan(f"S1_C_PD{k:06d}.dat", E)
        if E == 285.0:          # repeated later: 'latest full-range wins'
            k += 1
            pd_scan(f"S1_C_PD{k:06d}.dat", E)
    # energy scan at θ=4 (PD, mono scanned) and a dummy end-of-script file
    es = "S1_C_PD%06d.dat" % (k + 1)
    Es = np.linspace(250, 350, 201)
    _write(os.path.join(raw, es), t, [Es, 0.05 + 0.01 * np.sin(Es / 7), np.zeros(201),
                                      np.full(201, 500)])
    rows.append(_row(es, t, "pd", T["gain"], 350.0, 4.0))
    dm = "S1_C_PD%06d.dat" % (k + 2)
    _write(os.path.join(raw, dm), t, [[0.0, 0.9], [0.1, 0.1], [0, 0], [500, 500]])
    rows.append(_row(dm, t, "pd", T["gain"], 270.0, 1.0))
    t += timedelta(minutes=5)
    for j, E in enumerate(energies):
        ct_scan(f"S1_C_CT1{j + 1:06d}.dat", E, 9.0, 20.0, 101)
    for j, E in enumerate(energies):
        ct_scan(f"S1_C_CT2{j + 1:06d}.dat", E, 18.0, 32.0, 111,
                filt="empty" if E == 300.0 else "3 mm")
    # open beam (gain 8 -> reads 10x lower than the gain-9 sample PD)
    t0 = datetime(2025, 11, 15, 7, 30)
    Eo = np.linspace(250, 350, 501)
    _write(os.path.join(obd, "open_beam_carbon000001.dat"), t0,
           [Eo, np.full(501, T["i0"] / 10), np.zeros(501), np.full(501, 500)])
    orow = _row("open_beam_carbon000001.dat", t0, "pd", 8.0, 350.0, 0.0005, scan="1 axis")
    log = os.path.join(root, "Test.log")
    hdr = "\t".join(["met version", "log version", "filename"] + ["h"] * 40)
    with open(log, "w") as fh:
        fh.write("Tester, Test.log\n" + hdr + "\n" + orow + "\n" + "\n".join(rows) + "\n")
    return raw, dict(TRUTH, log=log, open_beam=obd)
