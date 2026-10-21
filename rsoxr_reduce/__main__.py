"""
python -m rsoxr_reduce <command> -p PROJECT ...

PROJECT is the output folder (default <raw>/ReducedAuto) holding reduction.json;
passing the raw folder also works once a project exists.

Commands:
    init        create a project for one raw folder (one edge/round)
    show        folder-wide decisions + last-run status counts
    inventory   classify every raw file (angle / energy scan / open beam / dummy ...)
    select      file choice per energy and section, with reasons for every file not used
    plan        proposed trims + reasons + flags (nothing written)
    reduce      plan + stitch + QC; writes .dat for ok/accepted energies, diagnostics,
                record, report, overviews
    review      energies held for review / rejected, with flags and PNG paths
    why E       every decision for one energy
    override E  change trims/background/scale/file/open beam for one energy (+ --reason)
    clear E     remove all overrides for one energy
    accept E    release a held energy (after reviewing its diagnostic)
    folder      set folder-wide clip angle / PD background
    set KEY VAL change a setting (e.g. trim.snr_min 4)
    escan       reduce energy scans (÷ open beam)
    waterfall   auto curves over a reference folder (--reference DIR), one waterfall
    compare DIR compare written curves with another reduced folder (e.g. manual Reduced/)
    calibrate   rebuild the manual-reduction ground truth for a beamtime folder
"""

import argparse
import json
import os
import sys


def _proj(a):
    from .project import ReductionProject
    return ReductionProject.open(a.project)


def _fmt_flags(flags, sev=("review", "reject")):
    return ", ".join(sorted({f["code"] for f in flags if f["severity"] in sev})) or "-"


def cmd_init(a):
    from .project import ReductionProject
    p = ReductionProject.create(a.raw, log=a.log, open_beams=a.open_beam, sample=a.sample,
                                out=a.out or a.project, description=a.description or "")
    files = p.files()
    counts = {}
    for f in files:
        counts[f.cls] = counts.get(f.cls, 0) + 1
    print(f"project {p.path}\n  sample {p.sample}\n  raw {p.cfg['raw']}\n  log {p.cfg['log']}")
    print(f"  open beams: {', '.join(o.name for o in p.open_beams()) or 'NONE FOUND'}")
    print("  files: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


def cmd_show(a):
    p = _proj(a)
    sel = p.selection()
    print(f"{p.sample}  raw={p.cfg['raw']}")
    print("sections: " + ", ".join(f"{s['id']} {s['detector']} {s['x0']:g}-{s['x1']:g}° "
                                   f"(at {s['support']:.0%} of energies)"
                                   for s in sel["sections"]))
    print(f"energies: {len(sel['energies'])}   energy scans: {len(sel['energy_scans'])}")
    lr = p.cfg.get("last_run")
    if lr:
        print(f"last run {lr['when']}: " + ", ".join(f"{k} {v}" for k, v in lr["counts"].items()))
        print(f"  clip angle {lr['clip_angle']}°, PD background "
              f"{lr['pd_background']['bg']:.5f} ({lr['pd_background']['source']})")
    ov = p.cfg["overrides"]
    if ov.get("energies") or ov.get("clip_angle") or ov.get("pd_background"):
        print("overrides: " + json.dumps({k: v for k, v in ov.items() if v}, default=str)[:600])


def cmd_inventory(a):
    p = _proj(a)
    for f in p.files():
        if a.all or f.cls != "angle":
            print(f"{f.name:40s} {f.cls:12s} {f.detector or '':4s} "
                  f"E={f.energy if f.energy is None else round(f.energy, 2)!s:8s} {f.reason}")
    counts = {}
    for f in p.files():
        counts[f.cls] = counts.get(f.cls, 0) + 1
    print("totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + ("" if a.all else "   (angle scans hidden; --all to list)"))


def cmd_select(a):
    p = _proj(a)
    sel = p.selection()
    for r in sel["energies"]:
        if a.energy and abs(r["energy"] - a.energy) > 0.06:
            continue
        print(f"{r['energy']:9.3f}  " + "  ".join(f"{k}={v}" for k, v in r["chosen"].items())
              + f"   flags: {_fmt_flags(r['flags'], ('review', 'reject', 'info'))}")
        for x in r["rejected"]:
            print(f"           not used: {x['file']} [{x['code']}] {x['reason']}")
    if not a.energy:
        print(f"never used ({len(sel['unused'])}): " +
              ", ".join(f"{u['file']}[{u['code']}]" for u in sel["unused"]))


def cmd_plan(a):
    p = _proj(a)
    plan = p.plan()
    print(f"clip angle {plan['clip_angle']}°; PD background {plan['pd_background']['bg']:.5f} "
          f"({plan['pd_background']['source']})")
    for pe in plan["energies"]:
        if a.energy and abs(pe["energy"] - a.energy) > 0.06:
            continue
        secs = "  ".join(f"{s['id']}[{s['start']},{s['end']}] {s['theta0']:.2f}-"
                         f"{s['theta1']:.2f}°" for s in pe["sections"])
        print(f"{pe['energy']:9.3f}  {secs or pe.get('reason', '')}   "
              f"flags: {_fmt_flags(pe['flags'])}")


def cmd_reduce(a):
    p = _proj(a)
    plan, out = p.run(write=True, diagnostics=a.diagnostics, energies=a.energies,
                      reference=a.reference)
    c = p.cfg["last_run"]["counts"]
    print(f"{p.sample}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print(f"  output: {p.path}")
    for pe in plan["energies"]:
        if pe["status"] in ("review", "rejected"):
            fl = out.get(pe["energy"], {}).get("all_flags", pe["flags"])
            print(f"  {pe['energy']:9.3f} {pe['status']:8s} {_fmt_flags(fl)}")
    print("  report: reduction_report.md   record: reduction_record.csv")


def cmd_review(a):
    p = _proj(a)
    lr = p.cfg.get("last_run")
    if not lr:
        sys.exit("no run yet: python -m rsoxr_reduce reduce -p ...")
    for k, v in lr["energies"].items():
        if v["status"] in ("review", "rejected") or (a.all and v["status"] == "accepted"):
            print(f"{float(k):9.3f} {v['status']:8s} {', '.join(v['review'])}")
            if v.get("diagnostic"):
                print(f"           {v['diagnostic']}")


def cmd_why(a):
    p = _proj(a)
    plan = p.plan()
    key, e = p.energy_key(a.energy)
    pe = next(x for x in plan["energies"] if abs(x["energy"] - e) < 1e-6)
    from . import pipeline as P
    sel = next(r for r in p.selection()["energies"] if abs(r["energy"] - e) < 1e-6)
    print(f"{p.sample} {e:.3f} eV")
    print(f"folder: clip angle {plan['clip_angle']}°, PD background "
          f"{plan['pd_background']['bg']:.5f} ({plan['pd_background']['source']})")
    for s in pe["sections"]:
        print(f"  {s['id']:4s} {s['file']}  trim ({s['start']}, {s['end']}) "
              f"keep {s['theta0']:.3f}-{s['theta1']:.3f}°  bg {s['bg']:.5f}")
        for w in s["reasons"]:
            print(f"        - {w}")
    for x in sel["rejected"]:
        print(f"  not used: {x['file']} [{x['code']}] {x['reason']}")
    print(f"  I0: {pe.get('open_beam_reason')}")
    if pe["sections"]:
        run = P.run_folder({**plan, "energies": [pe]}, p.settings)
        v = run.get(pe["energy"], {})
        for st in v.get("stats", []):
            print(f"  junction {st['pair']}: scale {st['scale']:.4g}, n={st['n']}, "
                  f"±{st.get('sigma', float('nan')):.1%}")
        flags = v.get("all_flags", pe["flags"])
    else:
        flags = pe["flags"]
        print(f"  {pe.get('reason', '')}")
    for f in flags:
        print(f"  [{f['severity']}] {f['code']}: {f['detail']}")
    print("  (cross-energy SCALE_OUTLIER checks need a full `reduce`)")


def cmd_override(a):
    p = _proj(a)
    e = p.override(a.energy, section=a.section, start=a.start, end=a.end, bg=a.bg,
                   scale=a.scale, drop=a.drop, use=a.use, open_beam=a.open_beam,
                   reason=a.reason)
    print(f"override stored for {e:.3f} eV; run `reduce` (or `why {e:g}`) to apply")


def cmd_clear(a):
    e = _proj(a).clear_override(a.energy)
    print(f"overrides cleared for {e:.3f} eV")


def cmd_accept(a):
    e = _proj(a).accept(a.energy, reason=a.reason)
    print(f"{e:.3f} eV accepted; run `reduce` to write it")


def cmd_folder(a):
    p = _proj(a)
    p.set_folder(clip_angle=a.clip_angle, pd_background=a.pd_background)
    print("folder overrides:", {k: v for k, v in p.cfg["overrides"].items() if k != "energies"})


def cmd_set(a):
    old, new = _proj(a).set_setting(a.key, a.value)
    print(f"{a.key}: {old} -> {new}")


def cmd_escan(a):
    p = _proj(a)
    res = p.energy_scans(write=True)
    if not res:
        print("no energy scans in this folder")
    for r in res:
        m = r["meta"]
        print(f"{m['file']}: θ={m.get('theta')} {m.get('erange')} -> "
              f"{r.get('path', 'not written')}   flags: "
              f"{', '.join(f['code'] for f in r['flags']) or '-'}")


def cmd_compare(a):
    p = _proj(a)
    rows = p.compare(a.reference)
    print(f"{'E':>9s} {'factor':>9s} {'shape med':>9s} {'shape max':>9s}  Q range (mine | ref)")
    for r in rows:
        if r["factor"] is None:
            continue
        q = r["q"]
        print(f"{r['energy']:9.2f} {r['factor']:9.3g} {r['shape_med']:9.1%} "
              f"{r['shape_max']:9.1%}  {q[0]:.4f}-{q[1]:.4f} | {q[2]:.4f}-{q[3]:.4f}")


def cmd_waterfall(a):
    """Overlay the written auto curves on a reference folder (e.g. manual Reduced/)."""
    import numpy as np
    from .calibrate import reduced_files
    from .plots import overview_waterfall_compare
    p = _proj(a)
    mine, refs = reduced_files(p.path), reduced_files(a.reference)
    status = {float(k): v["status"] for k, v in (p.cfg.get("last_run") or {}).get("energies", {}).items()}
    ents = [(e, np.loadtxt(f, skiprows=1, ndmin=2),
             status.get(min(status, key=lambda x: abs(x - e)), "ok") if status else "ok")
            for e, f in sorted(mine.items())]
    refd = {e: np.loadtxt(f, skiprows=1, ndmin=2) for e, f in refs.items()}
    out = a.out or os.path.join(p.path, "overview_waterfall_vs_reference.png")
    path, fac = overview_waterfall_compare(ents, refd, out, sample=p.sample,
                                           ref_label=a.label)
    print(f"{path}\n  reference × {fac:.3g} (median auto/reference ratio)")


def cmd_calibrate(a):
    from .calibrate import build_truth
    rows = build_truth(a.beamtime, log_path=a.log, out_csv=a.out)
    print(f"{len(rows)} section rows -> {a.out}")


def main(argv=None):
    # never crash printing Å/χ to a cp1252 console or pipe (Windows)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(prog="python -m rsoxr_reduce",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, proj=True, **kw):
        s = sub.add_parser(name, **kw)
        if proj:
            s.add_argument("-p", "--project", required=True)
        s.set_defaults(fn=fn)
        return s

    s = add("init", cmd_init, proj=False)
    s.add_argument("raw", help="raw data folder (one edge / round)")
    s.add_argument("-p", "--project", default=None, help="output/project folder")
    s.add_argument("--out", default=None)
    s.add_argument("--log", default=None)
    s.add_argument("--open-beam", nargs="+", default=None, help="open-beam files or folders")
    s.add_argument("--sample", default=None)
    s.add_argument("--description", default=None)

    add("show", cmd_show)
    s = add("inventory", cmd_inventory)
    s.add_argument("--all", action="store_true")
    s = add("select", cmd_select)
    s.add_argument("--energy", type=float, default=None)
    s = add("plan", cmd_plan)
    s.add_argument("--energy", type=float, default=None)
    s = add("reduce", cmd_reduce)
    s.add_argument("--energies", nargs="+", default=None)
    s.add_argument("--diagnostics", choices=["all", "flagged", "none"], default="flagged")
    s.add_argument("--reference", default=None, help="overlay curves from this folder")
    s = add("review", cmd_review)
    s.add_argument("--all", action="store_true")
    s = add("why", cmd_why)
    s.add_argument("energy", type=float)
    s = add("override", cmd_override)
    s.add_argument("energy", type=float)
    s.add_argument("--section", default=None, help="PD, CT1, CT2 ...")
    s.add_argument("--start", type=int, default=None)
    s.add_argument("--end", type=int, default=None, help="legacy convention: -1 drops last point")
    s.add_argument("--bg", type=float, default=None)
    s.add_argument("--scale", type=float, default=None)
    s.add_argument("--drop", default=None, help="section id or file name to leave out")
    s.add_argument("--use", default=None, help="file to use for --section")
    s.add_argument("--open-beam", default=None, help="open-beam file name")
    s.add_argument("--reason", required=True)
    s = add("clear", cmd_clear)
    s.add_argument("energy", type=float)
    s = add("accept", cmd_accept)
    s.add_argument("energy", type=float)
    s.add_argument("--reason", default="reviewed diagnostic")
    s = add("folder", cmd_folder)
    s.add_argument("--clip-angle", type=float, default=None)
    s.add_argument("--pd-background", type=float, default=None)
    s = add("set", cmd_set)
    s.add_argument("key")
    s.add_argument("value")
    add("escan", cmd_escan)
    s = add("compare", cmd_compare)
    s.add_argument("reference")
    s = add("waterfall", cmd_waterfall)
    s.add_argument("--reference", required=True, help="folder of reduced files to overlay")
    s.add_argument("--label", default="manual")
    s.add_argument("--out", default=None)
    s = add("calibrate", cmd_calibrate, proj=False)
    s.add_argument("beamtime", help="beamtime folder with <sample>/<edge>/Reduced outputs")
    s.add_argument("--log", default=None)
    s.add_argument("--out", required=True, help="CSV to write")

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
