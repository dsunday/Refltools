"""
energyscan.py
-------------
Reduce sample energy scans (fixed θ/2θ, mono scanned): subtract the PD dark
offset, divide point-by-point by the open beam interpolated onto the scan's
energies, correct the PD-gain difference exactly, flag what cannot be
corrected (slit/filter mismatch, extrapolation).

Output: <out>/EnergyScans/<sample>_Escan_th<θ>_<E0>-<E1>eV.dat
        columns  Energy(eV)  R  Error(= error_frac·R), '#' header with θ, 2θ,
        source file, open beam, background, gain factor.
"""

import os

import numpy as np

from .inventory import read_raw
from .pipeline import choose_open_beam
from .stitch import OpenBeam


def reduce_energy_scan(fi, open_beams, bg, settings):
    """fi: FileInfo (cls 'energy_scan'). -> dict(data, meta, flags)"""
    d = read_raw(fi.path)[1]
    d = d[np.argsort(d[:, 0])]
    E, y = d[:, 0], d[:, 1] - bg
    ref = {"datetime": fi.datetime, "pd_gain": fi.pd_gain, "slit_gap": fi.slit_gap,
           "filter": fi.filter}
    mid = float(np.median(E))
    ob, factor, why, flags = choose_open_beam(mid, ref, fi.edge, open_beams, settings)
    flags = [f for f in flags if f["code"] != "OB_EXTRAP"]
    if ob is None:
        return {"data": None, "meta": {"file": fi.name, "reason": why}, "flags": flags}
    O = OpenBeam(ob.path)
    lo, hi = O.erange
    outside = (E < lo - 0.1) | (E > hi + 0.1)      # ignore set-point jitter
    if outside.any():
        flags.append({"code": "OB_EXTRAP", "severity": "review",
                      "detail": f"{outside.sum()} of {len(E)} points outside {ob.name} "
                                f"({lo:.1f}-{hi:.1f} eV) extrapolated"})
    i0 = np.array([O(e) for e in E]) * factor
    R = y / i0
    nonpos = int((R <= 0).sum())
    if nonpos:
        flags.append({"code": "NEG_AFTER_BG", "severity": "info",
                      "detail": f"{nonpos} point(s) ≤ 0 after background"})
    ef = settings["output"]["error_frac"]
    data = np.column_stack([E, R, np.abs(R) * ef])
    th = fi.extra.get("theta")
    tth = fi.extra.get("two_theta")
    meta = {"file": fi.name, "theta": th, "two_theta": tth, "open_beam": ob.name,
            "open_beam_reason": why, "i0_factor": factor, "background": bg,
            "erange": [float(E.min()), float(E.max())], "datetime": fi.datetime}
    return {"data": data, "meta": meta, "flags": flags}


def write_energy_scan(out_dir, sample, result):
    m = result["meta"]
    os.makedirs(out_dir, exist_ok=True)
    th = m["theta"]
    name = (f"{sample}_Escan_th{th:g}_{m['erange'][0]:.0f}-{m['erange'][1]:.0f}eV.dat"
            if th is not None else
            f"{sample}_Escan_{os.path.splitext(m['file'])[0]}.dat")
    path = os.path.join(out_dir, name)
    hdr = "\n".join([
        f"# RSoXR energy scan reduced by rsoxr_reduce",
        f"# source: {m['file']}  ({m['datetime']})",
        f"# theta = {m['theta']} deg, two_theta = {m['two_theta']} deg",
        f"# open beam: {m['open_beam_reason']}",
        f"# PD background subtracted: {m['background']:.5f};  I0 gain factor: {m['i0_factor']:g}",
        "Energy(eV)\tR\tError",
    ])
    np.savetxt(path, result["data"], delimiter="\t", header=hdr, comments="")
    return path
