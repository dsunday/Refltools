"""
selection.py
------------
Group angle scans by energy, identify detector sections, and pick one file
per (energy, section) with an explicit reason for every file not used.

Sections
    Clustered per detector by angular-range overlap (IoU), independent of the
    filename and of the CEM count time (which changes with energy).  Clusters
    present at >= ``section_min_support`` of the energies are *nominal*
    sections; ordered by start angle -> stitch order (PD first).

Rules (reason codes)
    DUMMY / EMPTY / ZERO    (inventory) never used
    ENERGY_SCAN / OPEN_BEAM routed to energyscan.py
    ALIGNMENT / UNLOGGED    never used
    PARTIAL_RANGE           span < partial_frac * nominal span; used only if
                            nothing better exists (energy then flagged)
    SUPERSEDED              another full-range scan of the same section at
                            the same energy was taken later (latest wins)
    OFF_SECTION             range does not match any nominal section
    NO_PD        (reject)   no photodiode section: the energy cannot be normalised
    MISSING_SECTION (flag)  a middle section is absent (gap in Q) -> review
    TRUNCATED_Q  (info)     trailing section(s) absent: shorter Q range
    MIXED_SESSION   (flag)  chosen sections come from sessions > off_date_hours apart
    POSITION_MISMATCH       sections of one energy measured at different spots
                            (info; review above position_review_mm).  Spots are
                            often stepped on purpose to limit beam damage.
"""

from datetime import datetime

import numpy as np


def _iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def cluster_energies(files, tol):
    """-> list of (energy_label, [FileInfo]) sorted by energy."""
    fs = sorted(files, key=lambda f: f.energy)
    groups = []
    for f in fs:
        if groups and f.energy - groups[-1][-1].energy <= tol:
            groups[-1].append(f)
        else:
            groups.append([f])
    out = []
    for g in groups:
        pd = [f.energy for f in g if f.detector == "pd"]
        label = float(np.median(pd if pd else [f.energy for f in g]))
        out.append((label, g))
    return out


def build_sections(files, n_energies, settings):
    """Cluster angle files into sections. Returns (sections, assignment).

    sections: list of dicts {id, detector, x0, x1, support, nominal}
    assignment: {filename: section id}
    """
    g = settings["group"]
    clusters = []   # dicts with detector, members
    for f in sorted(files, key=lambda f: (f.detector or "", f.x0, -f.x1)):
        best, best_iou = None, 0.0
        for c in clusters:
            if c["detector"] != f.detector:
                continue
            iou = _iou(f.x0, f.x1, c["x0"], c["x1"])
            if iou > best_iou:
                best, best_iou = c, iou
        if best is not None and best_iou >= 0.6:
            best["members"].append(f)
            best["x0"] = float(np.median([m.x0 for m in best["members"]]))
            best["x1"] = float(np.median([m.x1 for m in best["members"]]))
        else:
            clusters.append({"detector": f.detector, "x0": f.x0, "x1": f.x1,
                             "members": [f]})
    for c in clusters:
        c["support"] = len({round(m.energy, 1) for m in c["members"]}) / max(n_energies, 1)
        c["nominal"] = c["support"] >= g["section_min_support"]
    nominal = sorted([c for c in clusters if c["nominal"]],
                     key=lambda c: (c["x0"], c["detector"] != "pd"))
    sections, assign = [], {}
    for i, c in enumerate(nominal):
        label = "PD" if c["detector"] == "pd" and i == 0 else (
            f"PD{i}" if c["detector"] == "pd" else f"CT{sum(1 for s in sections if s['detector'] == 'cem') + 1}")
        s = {"id": label, "detector": c["detector"], "x0": round(c["x0"], 4),
             "x1": round(c["x1"], 4), "support": round(c["support"], 3),
             "count_times": sorted({m.count_time for m in c["members"]
                                    if m.count_time is not None})}
        sections.append(s)
        for m in c["members"]:
            assign[m.name] = label
    # non-nominal clusters: attach to the same-detector section they overlap most
    for c in clusters:
        if c["nominal"]:
            continue
        for m in c["members"]:
            cand = [(max(0.0, min(m.x1, s["x1"]) - max(m.x0, s["x0"])), s)
                    for s in sections if s["detector"] == m.detector]
            cand = [t for t in cand if t[0] > 0]
            if cand:
                assign[m.name] = max(cand, key=lambda t: t[0])[1]["id"]
            else:
                assign[m.name] = None
    return sections, assign


def _dt(f):
    return datetime.fromisoformat(f.datetime) if f.datetime else datetime.min


def select(files, settings):
    """Full selection over one folder's inventory.

    Returns dict with 'sections', 'energies' (list of per-energy dicts),
    'unused' (files never considered, with reasons), 'energy_scans'.
    """
    g, s = settings["group"], settings["select"]
    unused, escans, openbeams = [], [], []
    angle = []
    for f in files:
        if f.cls == "angle":
            angle.append(f)
        elif f.cls == "energy_scan":
            escans.append(f)
        elif f.cls == "open_beam":
            openbeams.append(f)
        else:
            unused.append({"file": f.name, "code": f.cls.upper(), "reason": f.reason})
    egroups = cluster_energies(angle, g["energy_tol"])
    sections, assign = build_sections(angle, len(egroups), settings)
    sec_by_id = {x["id"]: x for x in sections}

    energies = []
    for E, fs in egroups:
        rec = {"energy": round(E, 4), "chosen": {}, "rejected": [], "flags": []}
        by_sec = {}
        for f in fs:
            sid = assign.get(f.name)
            if sid is None:
                rec["rejected"].append({"file": f.name, "code": "OFF_SECTION",
                                        "reason": f"range {f.x0:.2f}-{f.x1:.2f}° "
                                                  f"({f.detector}) matches no section"})
                continue
            by_sec.setdefault(sid, []).append(f)
        present = [x["id"] for x in sections if by_sec.get(x["id"])]
        for sec in sections:
            cands = by_sec.get(sec["id"], [])
            if not cands:
                k = [x["id"] for x in sections].index(sec["id"])
                later = any(x["id"] in present for x in sections[k + 1:])
                if sec["detector"] == "pd" and k == 0:
                    rec["flags"].append({"code": "NO_PD", "severity": "reject",
                                         "detail": "no photodiode scan: cannot normalise"})
                elif later:
                    rec["flags"].append({"code": "MISSING_SECTION", "severity": "review",
                                         "detail": f"no {sec['id']} scan ({sec['x0']:.1f}-"
                                                   f"{sec['x1']:.1f}°) between measured "
                                                   f"sections: gap in Q"})
                else:
                    rec["flags"].append({"code": "TRUNCATED_Q", "severity": "info",
                                         "detail": f"no {sec['id']} scan ({sec['x0']:.1f}-"
                                                   f"{sec['x1']:.1f}°): Q range ends early"})
                continue
            nom_span = sec["x1"] - sec["x0"]
            full = [f for f in cands if f.span >= s["partial_frac"] * nom_span]
            part = [f for f in cands if f not in full]
            pool = full if full else part
            pick = max(pool, key=lambda f: (_dt(f), f.counter))
            rec["chosen"][sec["id"]] = pick.name
            for f in cands:
                if f is pick:
                    continue
                if f in part:
                    rec["rejected"].append({
                        "file": f.name, "code": "PARTIAL_RANGE",
                        "reason": f"{f.x0:.2f}-{f.x1:.2f}° covers "
                                  f"{f.span / nom_span:.0%} of {sec['id']} "
                                  f"({sec['x0']:.1f}-{sec['x1']:.1f}°)"})
                else:
                    rec["rejected"].append({
                        "file": f.name, "code": "SUPERSEDED",
                        "reason": f"later full-range {sec['id']} scan {pick.name} "
                                  f"({pick.datetime}) replaces this one ({f.datetime})"})
            if not full:
                # a later start is harmless if it still overlaps the section
                # below; ending early (aborted scan) loses Q range -> review
                ends_early = pick.x1 < sec["x1"] - g["section_match_deg"]
                k = [x["id"] for x in sections].index(sec["id"])
                below = [by_sec[x["id"]] for x in sections[:k] if by_sec.get(x["id"])]
                gap = bool(below) and pick.x0 > max(f.x1 for f in below[-1])
                sev = "review" if (ends_early or gap) else "info"
                rec["flags"].append({"code": "PARTIAL_RANGE", "severity": sev,
                                     "detail": f"only a partial {sec['id']} scan: "
                                               f"{pick.name} {pick.x0:.2f}-{pick.x1:.2f}° "
                                               f"(nominal {sec['x0']:.1f}-{sec['x1']:.1f}°)"
                                               + ("; ends early" if ends_early else "")
                                               + ("; gap to the section below" if gap else "")})
        # sessions
        chosen = [f for f in fs if f.name in rec["chosen"].values()]
        times = sorted(_dt(f) for f in chosen if f.datetime)
        if len(times) > 1:
            span_h = (times[-1] - times[0]).total_seconds() / 3600
            if span_h > s["off_date_hours"]:
                rec["flags"].append({"code": "MIXED_SESSION", "severity": "review",
                                     "detail": f"chosen scans span {span_h:.1f} h: " +
                                     ", ".join(f"{f.name} {f.datetime}" for f in chosen)})
        rec["_files"] = {sid: next(f for f in fs if f.name == n)
                         for sid, n in rec["chosen"].items()}
        energies.append(rec)

    # sections of one energy must come from the same sample spot (the spot
    # itself is often stepped deliberately between energies)
    for r in energies:
        fs = list(r["_files"].values())
        xy = np.array([[f.x if f.x is not None else np.nan,
                        f.y if f.y is not None else np.nan] for f in fs])
        if len(fs) < 2 or not np.isfinite(xy).all():
            continue
        d = float(np.max(np.ptp(xy, axis=0)))
        if d > g["position_tol"]:
            sev = "review" if d > g["position_review_mm"] else "info"
            r["flags"].append({"code": "POSITION_MISMATCH", "severity": sev,
                               "detail": f"sections measured {d:.2f} mm apart: " +
                               ", ".join(f"{sid} ({f.x:.2f},{f.y:.2f})"
                                         for sid, f in r["_files"].items())})
    return {"sections": sections, "energies": energies, "unused": unused,
            "energy_scans": escans, "open_beams": openbeams}
