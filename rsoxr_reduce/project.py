"""
project.py
----------
ReductionProject: one raw-data folder (one edge / round) -> one output folder
that is also the project:

    <raw>/ReducedAuto/
        reduction.json              sources, settings, overrides, last run summary
        <sample>_<E>eV.dat          reduced curves (ok + accepted energies only)
        reduction_record.csv        one row per (energy, section): file, trims,
                                    reasons, background, scale, flags, status
        reduction_report.md         human summary (rejected files, held energies)
        angular_ranges.csv          legacy layout, one row per written energy
        diagnostics/<E>eV.png       per-energy diagnostic (all energies)
        overview_scales.png, overview_waterfall.png
        EnergyScans/                reduced energy scans

Existing ``Reduced/`` folders are never written to.
"""

import csv
import glob
import json
import os
import tempfile
from datetime import datetime

import numpy as np

from . import energyscan as ES
from . import inventory as I
from . import pipeline as P
from . import selection as SEL
from . import settings as S
from . import stitch as ST

SCHEMA_VERSION = 1


class ProjectError(RuntimeError):
    pass


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


class ReductionProject:
    def __init__(self, path, cfg):
        self.path = os.path.abspath(path)
        self.cfg = cfg
        self._log = None
        self._files = None
        self._sel = None
        self._obs = None

    # ------------------------------------------------------------------
    @classmethod
    def create(cls, raw, log=None, open_beams=None, sample=None, out=None,
               description=""):
        raw = os.path.abspath(raw)
        if not os.path.isdir(raw):
            raise ProjectError(f"{raw} is not a directory")
        log = os.path.abspath(log) if log else I.find_log(raw)
        if not log or not os.path.exists(log):
            raise ProjectError("no beamline .log found above the raw folder; pass --log")
        if not open_beams:
            # default: an OpenBeam* folder next to/above the raw folder, else
            # open-beam files anywhere in the log's folder
            cands = []
            d = raw
            for _ in range(4):
                cands += [p for p in glob.glob(os.path.join(d, "*"))
                          if os.path.isdir(p) and "open" in os.path.basename(p).lower()]
                d = os.path.dirname(d)
            open_beams = cands[:1] or [os.path.dirname(log)]
        out = os.path.abspath(out or os.path.join(raw, S.DEFAULTS["output"]["dir"]))
        if os.path.basename(out) == "Reduced":
            raise ProjectError("refusing to write into an existing manual 'Reduced/' folder")
        if os.path.exists(os.path.join(out, "reduction.json")):
            raise ProjectError(f"{out} already holds a reduction.json (use it, or pass --out)")
        if sample is None:
            # parent folder name, e.g. .../Brewer10/Carbon -> Brewer10
            sample = os.path.basename(os.path.dirname(raw)) or os.path.basename(raw)
        os.makedirs(out, exist_ok=True)
        cfg = {"schema_version": SCHEMA_VERSION, "sample": sample,
               "description": description,
               "created": datetime.now().isoformat(timespec="seconds"),
               "raw": raw, "log": log,
               "open_beams": [os.path.abspath(p) for p in open_beams],
               "settings": S.defaults(), "overrides": {"energies": {}},
               "last_run": None}
        p = cls(out, cfg)
        p.save()
        return p

    @classmethod
    def open(cls, path):
        path = os.path.abspath(path)
        fp = os.path.join(path, "reduction.json")
        if not os.path.exists(fp) and os.path.exists(
                os.path.join(path, S.DEFAULTS["output"]["dir"], "reduction.json")):
            path = os.path.join(path, S.DEFAULTS["output"]["dir"])
            fp = os.path.join(path, "reduction.json")
        if not os.path.exists(fp):
            raise ProjectError(f"no reduction.json in {path}")
        with open(fp) as fh:
            cfg = json.load(fh)
        # settings added in newer versions get their defaults
        cfg["settings"] = _deep_update(S.defaults(), cfg.get("settings", {}))
        return cls(path, cfg)

    def save(self):
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".reduction.", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.cfg, fh, indent=1, default=str)
        os.replace(tmp, os.path.join(self.path, "reduction.json"))

    # ------------------------------------------------------------------
    @property
    def settings(self):
        return self.cfg["settings"]

    @property
    def sample(self):
        return self.cfg["sample"]

    def set_setting(self, key, value):
        d = self.cfg["settings"]
        parts = key.split(".")
        for k in parts[:-1]:
            if k not in d:
                raise ProjectError(f"unknown setting {key}")
            d = d[k]
        if parts[-1] not in d:
            raise ProjectError(f"unknown setting {key}")
        old = d[parts[-1]]
        if isinstance(old, bool):
            value = str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(old, (int, float)) and not isinstance(value, (int, float)):
            value = type(old)(float(value))
        elif isinstance(old, dict) and not isinstance(value, dict):
            raise ProjectError(f"{key} is per-edge ({old}); set e.g. {key}.C")
        d[parts[-1]] = value
        self._sel = None
        self.save()
        return old, value

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    @property
    def log(self):
        if self._log is None:
            self._log = I.parse_log(self.cfg["log"])
        return self._log

    def files(self):
        if self._files is None:
            self._files = I.inventory(self.cfg["raw"], self.log, self.settings)
        return self._files

    def open_beams(self):
        if self._obs is None:
            self._obs = I.find_open_beams(self.cfg["open_beams"], self.log, self.settings)
        return self._obs

    def energy_key(self, energy):
        """Nearest planned energy (within 0.06 eV) -> canonical key string."""
        sel = self.selection()
        es = [r["energy"] for r in sel["energies"]]
        if not es:
            raise ProjectError("no energies")
        e = min(es, key=lambda x: abs(x - float(energy)))
        if abs(e - float(energy)) > 0.06:
            raise ProjectError(f"no energy near {energy} eV; have "
                               f"{', '.join(f'{x:g}' for x in es)}")
        return f"{e:.4f}", e

    def selection(self):
        if self._sel is None:
            sel = SEL.select(self.files(), self.settings)
            self._apply_file_overrides(sel)
            self._sel = sel
        return self._sel

    def _apply_file_overrides(self, sel):
        byname = {f.name: f for f in self.files()}
        for r in sel["energies"]:
            ov = self.cfg["overrides"]["energies"].get(f"{r['energy']:.4f}", {})
            for sid, fname in ov.get("use", {}).items():
                f = byname.get(fname)
                if f is None:
                    continue
                old = r["chosen"].get(sid)
                r["chosen"][sid] = fname
                r["_files"][sid] = f
                r["rejected"] = [x for x in r["rejected"] if x["file"] != fname]
                if old and old != fname:
                    r["rejected"].append({"file": old, "code": "USER_OVERRIDE",
                                          "reason": f"replaced by {fname}: "
                                                    f"{ov.get('use_reason', 'user')}"})
            # a user choice resolves the selection flags it addresses
            if ov.get("use"):
                r["flags"] = [x for x in r["flags"] if x["code"] not in
                              ("PARTIAL_RANGE", "MISSING_SECTION")
                              or not any(s in x["detail"] for s in ov["use"])]

    def plan(self):
        sel = self.selection()
        ov = dict(self.cfg["overrides"])
        return P.plan_folder(self.files(), sel, self.open_beams(), self.settings, ov)

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------
    def _ov(self, energy):
        key, e = self.energy_key(energy)
        return self.cfg["overrides"]["energies"].setdefault(key, {}), e

    def override(self, energy, section=None, start=None, end=None, bg=None, scale=None,
                 drop=None, use=None, open_beam=None, reason="user"):
        ov, e = self._ov(energy)
        if use:
            if not section:
                raise ProjectError("--use needs --section")
            ov.setdefault("use", {})[section] = use
            ov["use_reason"] = reason
            self._sel = None
        if drop:
            ov.setdefault("drop", []).append(drop)
        if open_beam:
            ov["open_beam"] = open_beam
        vals = {k: v for k, v in (("start", start), ("end", end), ("bg", bg),
                                  ("scale", scale)) if v is not None}
        if vals:
            if not section:
                raise ProjectError("trim/bg/scale overrides need --section")
            so = ov.setdefault("sections", {}).setdefault(section, {})
            so.update(vals)
            so["reason"] = reason
        ov.setdefault("log", []).append({"when": datetime.now().isoformat(timespec="seconds"),
                                         "section": section, "values": vals, "use": use,
                                         "drop": drop, "open_beam": open_beam,
                                         "reason": reason})
        self.save()
        return e

    def clear_override(self, energy):
        key, e = self.energy_key(energy)
        self.cfg["overrides"]["energies"].pop(key, None)
        self._sel = None
        self.save()
        return e

    def accept(self, energy, reason="reviewed"):
        ov, e = self._ov(energy)
        ov["accept"] = True
        ov.setdefault("log", []).append({"when": datetime.now().isoformat(timespec="seconds"),
                                         "accept": True, "reason": reason})
        self.save()
        return e

    def set_folder(self, clip_angle=None, pd_background=None):
        if clip_angle is not None:
            self.cfg["overrides"]["clip_angle"] = float(clip_angle)
        if pd_background is not None:
            self.cfg["overrides"]["pd_background"] = float(pd_background)
        self.save()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def out_name(self, energy):
        return self.settings["output"]["name"].format(sample=self.sample, energy=energy)

    def run(self, write=True, diagnostics="flagged", energies=None, reference=None):
        """Plan + reduce + QC.  Writes .dat for ok/accepted energies.

        diagnostics: 'all' | 'flagged' | 'none'.  reference: optional folder of
        existing reduced files to overlay on diagnostics (e.g. the manual
        ``Reduced/``).  Returns (plan, results).
        """
        from . import plots as PL
        plan = self.plan()
        if energies:
            want = [float(x) for x in energies]
            plan["energies"] = [pe for pe in plan["energies"]
                                if any(abs(pe["energy"] - w) < 0.06 for w in want)]
        out = P.run_folder(plan, self.settings)
        refs = {}
        if reference:
            from .calibrate import reduced_files
            refs = reduced_files(reference)
        ddir = os.path.join(self.path, "diagnostics")
        written = []
        for pe in plan["energies"]:
            v = out.get(pe["energy"])
            pe["output"] = None
            if not v:
                continue
            if write and pe["status"] in ("ok", "accepted"):
                fn = os.path.join(self.path, self.out_name(pe["energy"]))
                ST.write_dat(fn, v["res"]["q_data"])
                pe["output"] = fn
                written.append(pe)
            elif write:
                # a previously written file for an energy now held/rejected
                fn = os.path.join(self.path, self.out_name(pe["energy"]))
                if os.path.exists(fn):
                    os.replace(fn, fn + ".held")
            if diagnostics == "all" or (diagnostics == "flagged" and pe["status"] != "ok"):
                ref = None
                if refs:
                    k = min(refs, key=lambda x: abs(x - pe["energy"]))
                    if abs(k - pe["energy"]) < 0.06:
                        ref = (os.path.basename(reference.rstrip("/")),
                               np.loadtxt(refs[k], skiprows=1, ndmin=2))
                pe["diagnostic"] = PL.energy_diagnostic(
                    pe, v["res"], v["stats"], v["all_flags"],
                    os.path.join(ddir, f"{pe['energy']:.2f}eV.png"),
                    sample=self.sample, reference=ref)
        if write:
            self._write_record(plan, out)
            self._write_angular_ranges(written)
            self._write_report(plan, out)
            PL.overview_scales(plan, os.path.join(self.path, "overview_scales.png"),
                               sample=self.sample)
            PL.overview_waterfall(
                [(pe["energy"], out[pe["energy"]]["res"]["q_data"], pe["status"])
                 for pe in plan["energies"] if pe["energy"] in out],
                os.path.join(self.path, "overview_waterfall.png"), sample=self.sample)
            self.cfg["last_run"] = self._summary(plan, out)
            self.save()
        return plan, out

    def _summary(self, plan, out):
        ents = {}
        for pe in plan["energies"]:
            v = out.get(pe["energy"], {})
            ents[f"{pe['energy']:.4f}"] = {
                "status": pe["status"],
                "review": sorted({f["code"] for f in v.get("all_flags", pe["flags"])
                                  if f["severity"] in ("review", "reject")}),
                "sections": [{k: s[k] for k in ("id", "file", "start", "end", "theta0",
                                                "theta1", "bg")} for s in pe["sections"]],
                "output": pe.get("output"), "diagnostic": pe.get("diagnostic")}
        st = {}
        for x in ents.values():
            st[x["status"]] = st.get(x["status"], 0) + 1
        return {"when": datetime.now().isoformat(timespec="seconds"),
                "clip_angle": plan["clip_angle"], "pd_background": plan["pd_background"],
                "sections": plan["sections"], "counts": st, "energies": ents}

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def _write_record(self, plan, out):
        rows = []
        for pe in plan["energies"]:
            v = out.get(pe["energy"], {})
            flags = v.get("all_flags", pe["flags"])
            st = {s["index"]: s for s in v.get("stats", [])}
            for i, s in enumerate(pe["sections"]):
                j = st.get(i, {})
                rows.append({
                    "energy": pe["energy"], "status": pe["status"], "section": s["id"],
                    "file": s["file"], "start": s["start"], "end": s["end"],
                    "theta0": round(s["theta0"], 4), "theta1": round(s["theta1"], 4),
                    "background": s["bg"], "scale": j.get("scale", 1.0 if i == 0 else ""),
                    "overlap_n": j.get("n", ""), "scale_sigma": j.get("sigma", ""),
                    "open_beam": os.path.basename(pe.get("open_beam") or ""),
                    "i0_factor": pe.get("i0_factor", ""),
                    "reasons": " | ".join(s["reasons"]),
                    "flags": "; ".join(f"{f['severity']}:{f['code']}" for f in flags),
                })
            if not pe["sections"]:
                rows.append({"energy": pe["energy"], "status": pe["status"],
                             "reasons": pe.get("reason", ""),
                             "flags": "; ".join(f"{f['severity']}:{f['code']}"
                                                for f in pe["flags"])})
        keys = ["energy", "status", "section", "file", "start", "end", "theta0", "theta1",
                "background", "scale", "overlap_n", "scale_sigma", "open_beam",
                "i0_factor", "reasons", "flags"]
        with open(os.path.join(self.path, "reduction_record.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def _write_angular_ranges(self, written):
        n = max((len(pe["sections"]) for pe in written), default=0)
        hdr = ["Energy (eV)"] + [x for k in range(1, n + 1)
                                 for x in (f"Scan{k}_StartAngle", f"Scan{k}_EndAngle")]
        with open(os.path.join(self.path, "angular_ranges.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for pe in written:
                w.writerow([pe["energy"]] + [v for s in pe["sections"]
                                             for v in (s["theta0"], s["theta1"])])

    def _write_report(self, plan, out):
        L = [f"# Reduction report — {self.sample}",
             "", f"- raw: `{self.cfg['raw']}`", f"- log: `{self.cfg['log']}`",
             f"- run: {datetime.now().isoformat(timespec='seconds')}", ""]
        cnt = {}
        for pe in plan["energies"]:
            cnt[pe["status"]] = cnt.get(pe["status"], 0) + 1
        L.append("**Status:** " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
        L += ["", "## Folder-wide decisions", "",
              "| | value | how |", "|---|---|---|",
              f"| sections | {', '.join(s['id'] + ' ' + str(s['x0']) + '–' + str(s['x1']) + '°' for s in plan['sections'])} | clustered by detector + angular range |",
              f"| PD clip angle | {plan['clip_angle']} ° | median low-angle PD peak over energies |",
              f"| PD background | {plan['pd_background']['bg']:.5f} | {plan['pd_background']['source']} (n={plan['pd_background']['n']}) |",
              ""]
        held = [pe for pe in plan["energies"] if pe["status"] in ("review", "rejected")]
        L += ["## Energies held for review or rejected", ""]
        if not held:
            L.append("None.")
        for pe in held:
            v = out.get(pe["energy"], {})
            fl = [f for f in v.get("all_flags", pe["flags"]) if f["severity"] != "info"]
            L.append(f"- **{pe['energy']:.2f} eV — {pe['status']}**"
                     + (f" ({pe['reason']})" if pe.get("reason") else ""))
            for f in fl:
                L.append(f"  - `{f['code']}` {f['detail']}")
            if pe.get("diagnostic"):
                L.append(f"  - diagnostic: `{os.path.relpath(pe['diagnostic'], self.path)}`")
        sel = self.selection()
        L += ["", "## Files not used", "", "| file | code | reason |", "|---|---|---|"]
        for u in sel["unused"]:
            L.append(f"| {u['file']} | {u['code']} | {u['reason']} |")
        for pe in plan["energies"]:
            for x in pe.get("rejected_files", []):
                L.append(f"| {x['file']} | {x['code']} | {x['reason']} |")
        ovs = self.cfg["overrides"]
        if ovs.get("energies") or ovs.get("clip_angle") or ovs.get("pd_background"):
            L += ["", "## Overrides", "", "```", json.dumps(ovs, indent=1, default=str), "```"]
        with open(os.path.join(self.path, "reduction_report.md"), "w") as fh:
            fh.write("\n".join(L) + "\n")

    # ------------------------------------------------------------------
    def energy_scans(self, write=True):
        sel = self.selection()
        plan = self.plan()
        bg = plan["pd_background"]["bg"]
        res = []
        edir = os.path.join(self.path, "EnergyScans")
        for fi in sel["energy_scans"]:
            r = ES.reduce_energy_scan(fi, self.open_beams(), bg, self.settings)
            if write and r["data"] is not None:
                r["path"] = ES.write_energy_scan(edir, self.sample, r)
            res.append(r)
        return res

    def compare(self, reference):
        """Compare written curves with a reference folder of reduced files.
        -> list of dict(energy, factor, shape_med, shape_max, qmin/qmax both)."""
        from .calibrate import reduced_files
        refs = reduced_files(reference)
        mine = reduced_files(self.path)
        rows = []
        for e, f in sorted(mine.items()):
            if not refs:
                break
            k = min(refs, key=lambda x: abs(x - e))
            if abs(k - e) > 0.06:
                continue
            a = np.loadtxt(f, skiprows=1, ndmin=2)
            b = np.loadtxt(refs[k], skiprows=1, ndmin=2)
            rows.append(dict(energy=e, **compare_curves(a, b)))
        return rows


def compare_curves(a, b, nbins=30):
    """Binned log-ratio of two R(Q) curves on their common Q range.
    factor = median ratio a/b; shape_* = |log ratio| after removing it."""
    qa, ra, qb, rb = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    lo, hi = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    d = []
    for x0, x1 in zip(np.linspace(lo, hi, nbins)[:-1], np.linspace(lo, hi, nbins)[1:]):
        x = ra[(qa >= x0) & (qa < x1)]
        y = rb[(qb >= x0) & (qb < x1)]
        if len(x) and len(y) and np.median(x) > 0 and np.median(y) > 0:
            d.append(np.log(np.median(x) / np.median(y)))
    if not d:
        return {"factor": None, "shape_med": None, "shape_max": None}
    d = np.array(d)
    off = np.median(d)
    r = np.abs(d - off)
    return {"factor": float(np.exp(off)), "shape_med": float(np.exp(np.median(r)) - 1),
            "shape_max": float(np.exp(r.max()) - 1),
            "q": [float(qa.min()), float(qa.max()), float(qb.min()), float(qb.max())]}
