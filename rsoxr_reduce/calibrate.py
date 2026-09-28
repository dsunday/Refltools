"""
calibrate.py
------------
Build a ground-truth table of the manual reductions (via recover.py) and score
the automatic rules against it.

    truth = build_truth(beamtime_dir)          # one row per (folder, E, section)
    score = score_rules(truth_rows, auto_rows) # agreement statistics

Truth rows come from matching every point of every ``Reduced/*eV.dat`` back to
the raw files (exact to machine precision), so they record what was actually
done: file used, trim indices/angles, background, and slope a = scale/I0.
"""

import csv
import glob
import os
import re

import numpy as np

from . import inventory as I
from . import recover as REC
from . import selection as SEL

RED_RE = re.compile(r"_([\d.]+)eV\.dat$")


def edge_folders(beamtime_dir):
    """Folders holding raw angle scans + a manual Reduced output."""
    out = []
    for d in sorted(glob.glob(os.path.join(beamtime_dir, "*", "*", ""))):
        d = d.rstrip("/")
        if os.path.basename(d) == "Reduced":
            continue
        red = os.path.join(d, "Reduced")
        red = red if os.path.isdir(red) else d
        if glob.glob(os.path.join(red, "*eV.dat")):
            out.append((d, red))
    return out


def reduced_files(red_dir):
    out = {}
    for f in glob.glob(os.path.join(red_dir, "*eV.dat")):
        m = RED_RE.search(f)
        if m:
            out[float(m.group(1))] = f
    return out


def truth_for_folder(folder, red_dir, log, settings):
    fis = I.inventory(folder, log, settings)
    sel = SEL.select(fis, settings)
    reds = reduced_files(red_dir)
    angle = [f for f in fis if f.cls == "angle"]
    byname = {f.name: f for f in angle}
    sec_of = {}
    for r in sel["energies"]:
        for sid, n in r["chosen"].items():
            sec_of[n] = sid
    # section ids for superseded/partial files too
    _, assign = SEL.build_sections(angle, len(sel["energies"]), settings)
    rows = []
    for r in sel["energies"]:
        E = r["energy"]
        if not reds:
            break
        key = min(reds, key=lambda k: abs(k - E))
        if abs(key - E) > 0.06:
            continue
        cands = [f.path for f in angle if abs(f.energy - E) < 0.06]
        rec = REC.recover(reds[key], E, cands)
        a_pd = None
        for h in rec["sections"]:
            n = os.path.basename(h["file"])
            fi = byname[n]
            sid = assign.get(n)
            if sid == "PD" and a_pd is None:
                a_pd = h["a"]
            rows.append({
                "folder": os.path.relpath(folder, os.path.dirname(os.path.dirname(folder))),
                "edge": fi.edge, "energy": E, "section": sid, "file": n,
                "auto_choice": r["chosen"].get(sid) == n,
                "n_raw": h["n_raw"], "start": h["start"], "end": h["end"],
                "theta0": h["theta0"], "theta1": h["theta1"],
                "x0": fi.x0, "x1": fi.x1, "n_used": h["n_used"],
                "n_interior_missing": h["n_interior_missing"],
                "bg": h["bg"], "a": h["a"], "count_time": fi.count_time,
                "n_red": rec["n_red"], "explained": rec["explained"],
            })
        for row in rows:
            if row["energy"] == E and a_pd:
                row["scale_rel_pd"] = row["a"] / a_pd
    return rows


def build_truth(beamtime_dir, log_path=None, settings=None, out_csv=None, verbose=True):
    from .settings import defaults
    settings = settings or defaults()
    log = I.parse_log(log_path or I.find_log(beamtime_dir))
    rows = []
    for folder, red in edge_folders(beamtime_dir):
        try:
            rr = truth_for_folder(folder, red, log, settings)
        except Exception as e:  # keep going; report
            if verbose:
                print(f"  {folder}: FAILED {e}")
            continue
        if verbose:
            print(f"  {os.path.relpath(folder, beamtime_dir)}: {len(rr)} section rows")
        rows += rr
    if out_csv:
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def load_truth(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None
                continue
            try:
                r[k] = float(v) if any(c in v for c in ".e") or k not in (
                    "start", "end", "n_raw", "n_used") else int(v)
            except ValueError:
                if v in ("True", "False"):
                    r[k] = v == "True"
    return rows


def _num(x):
    return np.array([np.nan if v is None else float(v) for v in x])


def score_trims(truth, auto, tol=2):
    """Agreement of auto trims with truth, keyed by (folder, energy, file).

    auto: rows with keys folder, energy, file, start, end.
    Returns dict per section with fraction within ``tol`` points.
    """
    A = {(a["folder"], round(float(a["energy"]), 2), a["file"]): a for a in auto}
    out = {}
    for t in truth:
        k = (t["folder"], round(float(t["energy"]), 2), t["file"])
        a = A.get(k)
        if a is None:
            continue
        s = out.setdefault(t["section"], {"n": 0, "start_ok": 0, "end_ok": 0,
                                          "d_start": [], "d_end": []})
        s["n"] += 1
        ds = int(a["start"]) - int(t["start"])
        de = int(a["end"]) - int(t["end"])
        s["d_start"].append(ds)
        s["d_end"].append(de)
        s["start_ok"] += abs(ds) <= tol
        s["end_ok"] += abs(de) <= tol
    for s in out.values():
        s["start_frac"] = s["start_ok"] / s["n"] if s["n"] else None
        s["end_frac"] = s["end_ok"] / s["n"] if s["n"] else None
        s["d_start_med"] = float(np.median(s["d_start"])) if s["d_start"] else None
        s["d_end_med"] = float(np.median(s["d_end"])) if s["d_end"] else None
    return out
