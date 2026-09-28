"""
pipeline.py
-----------
Folder-level orchestration: selection -> folder-wide quantities (clip angle,
PD background) -> per-energy trims with reasons -> stitch -> QC.

    plan = plan_folder(files, sel, open_beams, settings, overrides)
    res  = reduce_energy(plan_entry, settings)

A *plan entry* is JSON-serialisable (stored in reduction.json) so the user or
Claude can inspect and override any decision before writing output.
"""

from datetime import datetime

import numpy as np

from . import qc as QC
from . import stitch as ST
from . import trim as TR
from .inventory import read_raw


def _load(path, cache):
    if path not in cache:
        cache[path] = read_raw(path)[1]
    return cache[path]


def choose_open_beam(energy, ref, edge, open_beams, settings):
    """Pick the open beam for one energy.

    ``ref`` is the FileInfo-like dict of the energy's PD scan (datetime,
    pd_gain, slit_gap, filter).  Candidates must be the same edge; they are
    ranked by (same filter, same exit-slit gap, covers the energy, time
    distance).  A PD-gain difference is corrected exactly (gain is in decades:
    I0_eff = I0 · 10^(gain_sample − gain_ob)); slit/filter differences cannot
    be corrected and are flagged.

    -> (FileInfo | None, i0_factor, reason, flags)
    """
    o = settings["open_beam"]
    same = [b for b in open_beams if b.edge == edge]
    if not same:
        return None, 1.0, "no open beam available", [
            {"code": "NO_OPEN_BEAM", "severity": "review",
             "detail": "no open-beam scan for this edge; curve normalised to its maximum"}]
    when = datetime.fromisoformat(ref["datetime"]) if ref.get("datetime") else None

    def slit_ok(b):
        if not (b.slit_gap and ref.get("slit_gap")):
            return True
        return abs(b.slit_gap / ref["slit_gap"] - 1) <= o["slit_tol"]

    def filt_ok(b):
        return not (b.filter and ref.get("filter")) or b.filter == ref["filter"]

    def dt(b):
        if not (b.datetime and when):
            return 1e12
        return abs((datetime.fromisoformat(b.datetime) - when).total_seconds())

    def key(b):
        return (not filt_ok(b), not slit_ok(b), not (b.x0 <= energy <= b.x1), dt(b))

    ob = min(same, key=key)
    flags = []
    factor = 1.0
    gs, gb = ref.get("pd_gain"), ob.pd_gain
    notes = []
    if gs is not None and gb is not None and gs != gb:
        if o["gain_correct"]:
            factor = 10.0 ** (gs - gb)
            flags.append({"code": "GAIN_CORRECTED", "severity": "info",
                          "detail": f"PD gain {gs:g} (sample) vs {gb:g} ({ob.name}); "
                                    f"I0 × {factor:g}"})
            notes.append(f"gain {gb:g}→{gs:g} corrected ×{factor:g}")
        else:
            flags.append({"code": "GAIN_MISMATCH", "severity": "review",
                          "detail": f"PD gain {gs:g} (sample) vs {gb:g} ({ob.name}) "
                                    f"NOT corrected: R off by 10^{gs - gb:g}"})
    if not slit_ok(ob):
        flags.append({"code": "SLIT_MISMATCH", "severity": "review",
                      "detail": f"exit slit {ref.get('slit_gap')} (sample) vs "
                                f"{ob.slit_gap} ({ob.name}): absolute R scale off by "
                                f"≈{ob.slit_gap / ref['slit_gap']:.2f}"})
    if not filt_ok(ob):
        flags.append({"code": "FILTER_MISMATCH", "severity": "review",
                      "detail": f"filter '{ref.get('filter')}' vs '{ob.filter}' ({ob.name})"})
    if not (ob.x0 <= energy <= ob.x1):
        flags.append({"code": "OB_EXTRAP", "severity": "review",
                      "detail": f"{energy:.2f} eV outside {ob.name} "
                                f"({ob.x0:.1f}-{ob.x1:.1f} eV); extrapolating"})
    why = (f"{ob.name}: {edge} edge, slit {ob.slit_gap} vs {ref.get('slit_gap')}, "
           f"filter '{ob.filter}', {dt(ob) / 3600:.1f} h from the PD scan"
           + (f"; {'; '.join(notes)}" if notes else ""))
    return ob, factor, why, flags


def plan_folder(files, sel, open_beams, settings, overrides=None, edge=None):
    """Build the per-energy plan. Returns dict(folder=..., energies=[...])."""
    overrides = overrides or {}
    cache = {}
    sections = sel["sections"]
    sec_kind = {s["id"]: ("pd" if s["detector"] == "pd" else "cem") for s in sections}
    energies = sel["energies"]
    if edge is None:
        edges = [f.edge for r in energies for f in r["_files"].values() if f.edge]
        edge = max(set(edges), key=edges.count) if edges else None

    # ---- folder-wide: clip angle -------------------------------------------
    pd_id = next((s["id"] for s in sections if s["detector"] == "pd"), None)
    pd_arrays = {}
    for r in energies:
        f = r["_files"].get(pd_id)
        if f is not None and f.x0 < 0.1:          # PD scans that start at 0
            pd_arrays[r["energy"]] = _load(f.path, cache)
    clip_angle, peaks = TR.folder_clip_angle(pd_arrays, edge, settings)
    if "clip_angle" in overrides:
        clip_angle = float(overrides["clip_angle"])

    # ---- folder-wide: PD background ----------------------------------------
    nxt = {s["id"]: (sections[i + 1]["id"] if i + 1 < len(sections) else None)
           for i, s in enumerate(sections)}
    fits = []
    if pd_id and nxt.get(pd_id):
        ct_id = nxt[pd_id]
        for r in energies:
            fp, fc = r["_files"].get(pd_id), r["_files"].get(ct_id)
            if fp is None or fc is None or sec_kind[ct_id] != "cem":
                continue
            fits.append(TR.fit_pd_background(_load(fp.path, cache), _load(fc.path, cache),
                                             fc.count_time, settings))
    bgi = TR.folder_background(fits, edge, settings)
    if "pd_background" in overrides:
        bgi = dict(bgi, bg=float(overrides["pd_background"]), source="override")

    folder = {"edge": edge, "clip_angle": clip_angle, "clip_peaks": peaks,
              "pd_background": bgi, "sections": sections}

    out = []
    for r in energies:
        out.append(plan_energy(r, folder, open_beams, settings,
                               overrides.get("energies", {}).get(f"{r['energy']:.4f}", {}),
                               cache))
    folder["energies"] = out
    return folder


def plan_energy(r, folder, open_beams, settings, ov, cache):
    """Trims and reasons for one energy (``r`` from selection)."""
    E = r["energy"]
    sections = folder["sections"]
    bgi = folder["pd_background"]
    flags = list(r["flags"])
    # apply file overrides
    files = dict(r["_files"])
    order = [s["id"] for s in sections if s["id"] in files]
    secs = []
    for sid in order:
        f = files[sid]
        kind = "pd" if f.detector == "pd" else "cem"
        secs.append({"id": sid, "file": f.name, "path": f.path, "kind": kind,
                     "count_time": f.count_time, "x0": f.x0, "x1": f.x1,
                     "bg": bgi["bg"] if kind == "pd" else 0.0,
                     "sigma": bgi["sigma"] if kind == "pd" else None,
                     "datetime": f.datetime, "filter": f.filter, "pd_gain": f.pd_gain,
                     "reasons": []})
    for sid in ov.get("drop", []):
        secs = [s for s in secs if s["id"] != sid and s["file"] != sid]
    if not secs or secs[0]["kind"] != "pd":
        return {"energy": E, "status": "rejected", "sections": [], "flags": flags,
                "rejected_files": r["rejected"],
                "reason": "no usable sections" if not secs else
                          "no photodiode section: cannot normalise to the open beam"}
    # start indices
    for s in secs:
        d = _load(s["path"], cache)
        s["n"] = len(d)
        if s["kind"] == "pd" and s["x0"] < 0.1:
            own, _ = TR.pd_peak(d, folder["edge"], settings)
            st, why, fl = TR.pd_start(d, folder["clip_angle"], own, settings)
        elif s["kind"] == "cem":
            st, why, fl = TR.ct_start({"id": s["id"], "data": d}, settings)
        else:
            st, why, fl = 0, "starts above the clipping range", []
        s["start"], s["reasons"] = int(st), [why]
        flags += fl
    # ends
    junction = []
    for i, s in enumerate(secs):
        d = _load(s["path"], cache)
        lo = dict(s, data=d)
        if i + 1 < len(secs):
            h = secs[i + 1]
            hi = dict(h, data=_load(h["path"], cache))
            last, why, fl, stt = TR.junction_end(lo, hi, s["start"], h["start"], settings)
            junction.append(stt)
        else:
            last, why, fl = TR.tail_end(lo, s["start"], settings)
        s["end"] = int(TR.legacy_end(last, len(d)))
        s["reasons"].append(why)
        s["theta0"] = float(d[s["start"], 0])
        s["theta1"] = float(d[last, 0])
        flags += fl
    # manual overrides of trims / background
    trim_codes = ("NO_OVERLAP", "LOW_OVERLAP", "NOISY_OVERLAP", "CLIP_SHIFT",
                  "LOW_COUNTS_TAIL", "CT_SATURATION")
    for s in secs:
        o = ov.get("sections", {}).get(s["id"], {})
        for k in ("start", "end", "bg", "scale"):
            if k in o:
                s[k] = o[k]
                s["reasons"].append(f"OVERRIDE {k}={o[k]}: {o.get('reason', 'user')}")
        if "start" in o or "end" in o:
            d = _load(s["path"], cache)
            s["theta0"] = float(d[s["start"], 0])
            s["theta1"] = float(d[TR.last_kept(len(d), s["end"]), 0])
            # automatic trim flags about this section no longer apply; QC re-checks
            flags = [f for f in flags if not (f["code"] in trim_codes
                                              and s["id"] in f["detail"])]
    # open beam, matched to the first (PD) section's acquisition settings
    f0 = files[secs[0]["id"]]
    ref = {"datetime": f0.datetime, "pd_gain": f0.pd_gain, "slit_gap": f0.slit_gap,
           "filter": f0.filter}
    if ov.get("open_beam"):
        ob = next((b for b in open_beams if b.name == ov["open_beam"]), None)
        _, factor, ob_why, fl = choose_open_beam(E, ref, folder["edge"], [ob], settings)
        ob_why = f"OVERRIDE: {ob_why}"
    else:
        ob, factor, ob_why, fl = choose_open_beam(E, ref, folder["edge"], open_beams, settings)
    flags += fl
    return {"energy": E, "sections": secs, "junctions": junction,
            "open_beam": ob.path if ob else None, "open_beam_reason": ob_why,
            "i0_factor": factor,
            "flags": flags, "rejected_files": r["rejected"],
            "status": "planned", "accepted": bool(ov.get("accept", False))}


def reduce_energy(pe, settings, cache=None, scale_overrides=None):
    """Stitch one plan entry. -> (result dict, junction stats, flags)"""
    cache = cache if cache is not None else {}
    secs = pe["sections"]
    scans = [_load(s["path"], cache) for s in secs]
    trims = [(s["start"], s["end"]) for s in secs]
    bgs = [s["bg"] for s in secs]
    i0 = None
    if pe.get("open_beam"):
        i0 = ST.OpenBeam(pe["open_beam"])(pe["energy"]) * pe.get("i0_factor", 1.0)
    so = dict(scale_overrides or {})
    for i, s in enumerate(secs):
        if s.get("scale") is not None:
            so[i] = s["scale"]
    res = ST.reduce_sections(scans, trims, bgs, pe["energy"], i0=i0,
                             error_frac=settings["output"]["error_frac"],
                             scale_overrides=so)
    res["i0"] = i0
    stats = QC.junction_stats(pe, res)
    # scale is PD volts per CEM rate: remove the PD gain decade so energies
    # measured at different gains share one trend
    g = secs[0].get("pd_gain")
    for st in stats:
        st["scale_n"] = st["scale"] * (10.0 ** (-g) if g is not None else 1.0)
        st["gain"] = g
    flags = QC.energy_flags(pe, res, stats, settings)
    flags += QC.filter_checks(pe, res, stats, i0, settings)
    return res, stats, flags


def run_folder(plan, settings):
    """Reduce every planned energy, add cross-energy QC, set status.

    Returns {energy: dict(result, stats, flags, status)} and mutates plan
    entries (flags, status, junction stats) in place.
    """
    cache = {}
    out = {}
    for pe in plan["energies"]:
        if not pe["sections"]:
            continue
        res, stats, fl = reduce_energy(pe, settings, cache)
        out[pe["energy"]] = {"res": res, "stats": stats, "flags": fl}
    per = sorted(((E, v["stats"]) for E, v in out.items()), key=lambda t: t[0])
    tflags, trends = QC.trend_flags(per, settings)
    # junctions without overlap: bridge the gap, else borrow from the nearest
    # energies that had overlap (always flagged for review)
    if settings["qc"]["use_trend_for_no_overlap"]:
        for pe in plan["energies"]:
            v = out.get(pe["energy"])
            if not v or not any(st["n"] == 0 and not st["override"] for st in v["stats"]):
                continue
            so, notes = {}, []
            for st in v["stats"]:
                if st["n"] > 0 or st["override"]:
                    continue
                i = st["index"]
                secs = v["res"]["sections"]
                prev = np.column_stack([np.concatenate([x["theta"] for x in secs[:i]]),
                                        np.concatenate([x["y"] for x in secs[:i]])])
                cur = np.column_stack([secs[i]["theta"], secs[i]["y"]])
                # y already includes ÷I0; scale is defined on I0-free data, and
                # I0 is common to all sections, so the ratio is the same
                sc, gap = QC.gap_bridge_scale(prev, cur, m=settings["qc"]["gap_points"],
                                              max_gap=settings["qc"]["max_gap_deg"])
                if sc is not None:
                    so[i] = sc
                    notes.append({"code": "SCALE_FROM_GAP", "severity": "review",
                                  "detail": f"{st['pair']}: no overlap ({gap:.3f}° gap); scale "
                                            f"{sc:.3g} from log-linear extrapolation across "
                                            f"the gap"})
                    continue
                sc, dists = QC.nearest_scale(per, pe["energy"], st["pair"])
                if sc is not None and st.get("gain") is not None:
                    sc = sc * 10.0 ** st["gain"]
                if sc is not None:
                    so[i] = sc
                    notes.append({"code": "SCALE_FROM_TREND", "severity": "review",
                                  "detail": f"{st['pair']}: no overlap; scale {sc:.3g} = median "
                                            f"of energies {max(dists):.1f} eV or closer"})
            if so:
                res, stats, fl = reduce_energy(pe, settings, cache, scale_overrides=so)
                fl = [f for f in fl if f["code"] != "NO_OVERLAP"] + notes
                out[pe["energy"]] = {"res": res, "stats": stats, "flags": fl}
    # noisy junctions: inverse-variance blend of the own overlap scale with the
    # neighbour trend (the PD/CEM sensitivity ratio varies smoothly with energy)
    if settings["qc"]["blend_noisy_scales"]:
        for pe in plan["energies"]:
            v = out.get(pe["energy"])
            if not v:
                continue
            so, notes = {}, []
            for st in v["stats"]:
                sig, tr = st.get("sigma"), st.get("trend")
                if (st["n"] == 0 or st["override"] or sig is None or tr is None
                        or sig <= settings["qc"]["noisy_overlap"]):
                    continue
                ts = st["trend_sigma"]
                w1, w2 = 1 / sig ** 2, 1 / ts ** 2
                ls = (w1 * np.log(st["scale"]) + w2 * np.log(tr)) / (w1 + w2)
                so[st["index"]] = float(np.exp(ls))
                notes.append({"code": "SCALE_BLENDED", "severity": "info",
                              "detail": f"{st['pair']}: own scale {st['scale']:.3g} ±{sig:.0%} "
                                        f"blended with neighbour trend {tr:.3g} ±{ts:.0%} "
                                        f"-> {np.exp(ls):.3g}"})
            if so:
                old = {s_["index"]: s_ for s_ in v["stats"]}
                res, stats, fl = reduce_energy(pe, settings, cache, scale_overrides=so)
                for s_ in stats:        # keep the pre-blend numbers for the record
                    o = old.get(s_["index"], {})
                    for k in ("trend", "trend_sigma", "trend_dev"):
                        if k in o:
                            s_[k] = o[k]
                    if s_["index"] in so:
                        s_["own_scale"], s_["own_sigma"] = o["scale"], o.get("sigma")
                        s_["n"], s_["sigma"] = o["n"], o.get("sigma")
                        s_["override"] = False
                out[pe["energy"]] = {"res": res, "stats": stats,
                                     "flags": [f for f in fl] + notes + [
                                         f for f in v["flags"] if f["code"] in
                                         ("SCALE_FROM_GAP", "SCALE_FROM_TREND")]}
    for pe in plan["energies"]:
        v = out.get(pe["energy"])
        if not v:
            pe["status"] = "rejected"
            continue
        allf = pe["flags"] + v["flags"] + tflags.get(pe["energy"], [])
        # de-duplicate identical code+detail
        seen, uniq = set(), []
        for f in allf:
            k = (f["code"], f["detail"])
            if k not in seen:
                seen.add(k)
                uniq.append(f)
        pe["qc_flags"] = [f for f in uniq if f not in pe["flags"]]
        pe["junctions_stitched"] = v["stats"]
        review = [f for f in uniq if f["severity"] == "review"]
        if any(f["severity"] == "reject" for f in uniq):
            pe["status"] = "rejected"
        else:
            pe["status"] = "ok" if not review else ("accepted" if pe.get("accepted") else "review")
        v["all_flags"] = uniq
    return out
