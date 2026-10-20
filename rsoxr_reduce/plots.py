"""
plots.py
--------
Headless (Agg) diagnostic PNGs, each with a .json sidecar describing it.

    energy_diagnostic   one energy: raw sections with trimmed regions shaded,
                        stitched R(Q) (+ optional reference curve), overlap ratios
    overview_waterfall  every reduced energy, offset
    overview_scales     junction scale factors vs energy (+ trend, flags),
                        kept angular ranges vs energy
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .inventory import read_raw  # noqa: E402

# categorical slots in fixed order (section identity): PD, CT1, CT2, CT3, ...
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
REVIEW, REJECT = "#c98500", "#d03b2f"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, out, meta):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    meta = dict(meta, created=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    return out


def energy_diagnostic(pe, res, stats, flags, out, sample="", reference=None):
    """reference: optional (label, q_data) curve overlaid, scaled to match."""
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.18)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, 0])
    a4 = fig.add_subplot(gs[1, 1])
    secs = pe["sections"]

    # (a) raw sections, trimmed parts faint + shaded
    for i, s in enumerate(secs):
        c = SERIES[i % len(SERIES)]
        d = read_raw(s["path"])[1]
        n = len(d)
        e = n + s["end"] if s["end"] < 0 else s["end"]
        y = d[:, 1] if s["kind"] == "pd" else d[:, 1] * (res["sections"][i]["scale"]
                                                         if i else 1.0)
        pos = y > 0
        a1.plot(d[pos, 0], y[pos], "-", color=c, lw=0.8, alpha=0.3)
        k = np.zeros(n, bool)
        k[s["start"]:e] = True
        a1.plot(d[k & pos, 0], y[k & pos], "-", color=c, lw=1.6,
                label=f"{s['id']}  {s['file']}")
        for lo, hi in ((d[0, 0], d[max(s['start'] - 1, 0), 0] if s["start"] else None),
                       (d[min(e, n - 1), 0] if e < n else None, d[-1, 0])):
            if lo is not None and hi is not None and hi > lo:
                a1.axvspan(lo, hi, color=c, alpha=0.07, lw=0)
    if secs and secs[0]["kind"] == "pd" and secs[0]["bg"] > 0:   # log axis
        a1.axhline(secs[0]["bg"], color=INK2, lw=0.8, ls="--")
        a1.text(secs[0].get("theta0", 0), secs[0]["bg"],
                f" PD background {secs[0]['bg']:.4f}", color=INK2, fontsize=7, va="bottom")
    a1.set_yscale("log")
    a1.set_xlabel("θ (deg)")
    a1.set_ylabel("raw signal (CT × stitch scale)")
    a1.set_title("sections: bold = kept, faint/shaded = trimmed", fontsize=9, color=INK)
    a1.legend(fontsize=7, frameon=False, loc="upper right")
    _style(a1)

    # (b) stitched curve
    q, r = res["q_data"][:, 0], res["q_data"][:, 1]
    th = res["theta_data"][:, 0]
    bounds = np.cumsum([0] + [x["n_kept"] for x in res["sections"]])
    from .stitch import to_q
    for i in range(len(res["sections"])):
        seg = res["theta_data"][bounds[i]:bounds[i + 1]]
        a2.plot(to_q(seg[:, 0], pe["energy"]), seg[:, 1], ".", ms=3,
                color=SERIES[i % len(SERIES)], label=secs[i]["id"])
    if reference is not None:
        lab, ref = reference
        m = (ref[:, 1] > 0)
        # scale the reference onto this curve by the median ratio on common Q
        qi = np.clip(np.searchsorted(np.sort(q), ref[m, 0]), 0, len(q) - 1)
        rr = np.interp(ref[m, 0], np.sort(q), r[np.argsort(q)])
        f = np.median(rr / ref[m, 1]) if len(qi) else 1.0
        a2.plot(ref[m, 0], ref[m, 1] * f, "-", color=INK2, lw=0.8, alpha=0.8,
                label=f"{lab} × {f:.3g}")
    a2.set_yscale("log")
    a2.set_xlabel("Q (Å⁻¹)")
    a2.set_ylabel("R")
    a2.set_title("stitched reflectivity", fontsize=9, color=INK)
    a2.legend(fontsize=7, frameon=False)
    _style(a2)
    _ = th

    # (c) overlap ratios per junction
    for i, info in enumerate(res["sections"]):
        if i == 0 or not info["ratios"]:
            continue
        rr = np.asarray(info["ratios"]) / info["scale"]
        c = SERIES[i % len(SERIES)]
        x = np.linspace(0, 1, len(rr)) if len(rr) > 1 else np.array([0.5])
        a3.plot(x, rr, "o-", color=c, ms=4, lw=1,
                label=f"{stats[i - 1]['pair']}: n={len(rr)}, scale ±"
                      f"{stats[i - 1].get('sigma', float('nan')):.0%}")
    a3.axhline(1, color=INK2, lw=0.8)
    a3.set_xlabel("position in overlap (low θ → high θ)")
    a3.set_ylabel("ratio / median")
    a3.set_title("overlap agreement per junction", fontsize=9, color=INK)
    if a3.get_legend_handles_labels()[0]:
        a3.legend(fontsize=7, frameon=False)
    _style(a3)

    # (d) decisions + flags as text
    a4.axis("off")
    lines = []
    for s in secs:
        lines.append(f"{s['id']}: keep {s['theta0']:.2f}–{s['theta1']:.2f}°  "
                     f"trim ({s['start']}, {s['end']})")
        for why in s["reasons"]:
            lines.append("    · " + why)
    lines.append(f"I0: {pe.get('open_beam_reason', '')}")
    lines.append("")
    for f in flags:
        mark = {"review": "▲ REVIEW", "reject": "✖ REJECT"}.get(f["severity"], "· info")
        lines.append(f"{mark} {f['code']}: {f['detail']}")
    import textwrap
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 105, subsequent_indent="      ") or [""]
    a4.text(0, 1, "\n".join(wrapped[:34]), va="top", ha="left", fontsize=6.6,
            family="monospace", color=INK, transform=a4.transAxes)

    col = {"review": REVIEW, "rejected": REJECT}.get(pe.get("status"), INK)
    fig.suptitle(f"{sample}  {pe['energy']:.2f} eV  —  {pe.get('status', '').upper()}",
                 color=col, fontsize=12, x=0.02, ha="left")
    return _save(fig, out, {"kind": "energy_diagnostic", "energy": pe["energy"],
                            "status": pe.get("status"),
                            "flags": [f["code"] for f in flags]})


def overview_waterfall(entries, out, sample="", decade_step=1.0):
    """entries: list of (energy, q_data, status) sorted by energy."""
    n = len(entries)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.28 * n + 2)))
    for k, (E, qd, status) in enumerate(entries):
        off = 10 ** (decade_step * k)
        m = qd[:, 1] > 0
        if not m.any():         # nothing positive to plot (e.g. a bad I0)
            continue
        c = {"review": REVIEW, "rejected": REJECT}.get(status, SERIES[0])
        ax.plot(qd[m, 0], qd[m, 1] * off, "-", lw=0.9, color=c)
        ax.text(qd[m, 0].max() * 1.01, qd[m, 1][-1] * off, f"{E:.1f}", fontsize=6,
                color=INK if status in ("ok", "accepted") else c, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel(f"R × 10^(k·{decade_step:g})")
    ax.set_title(f"{sample}: all energies (orange = held for review, red = rejected)",
                 fontsize=9, color=INK)
    _style(ax)
    return _save(fig, out, {"kind": "overview_waterfall", "n": n})


def overview_waterfall_compare(entries, reference, out, sample="", ref_label="manual",
                               decade_step=1.0):
    """Auto curves (lines) over reference curves (markers + thin lines).

    entries: list of (energy, q_data, status) for the auto reduction.
    reference: {energy: q_data} (e.g. the manual Reduced/ files).
    The reference is multiplied by ONE factor for the whole folder (median of
    the per-energy auto/reference ratios) so a global normalisation offset
    (e.g. the gain-8 open beam) does not hide shape differences; per-energy
    level differences stay visible.  Returns (path, factor).
    """
    from .project import compare_curves
    pairs = []
    for E, qd, status in entries:
        if not reference:
            break
        k = min(reference, key=lambda x: abs(x - E))
        if abs(k - E) < 0.06:
            pairs.append((E, qd, status, reference[k]))
    facs = [compare_curves(qd, ref)["factor"] for _, qd, _, ref in pairs]
    facs = [f for f in facs if f]
    fac = float(np.median(facs)) if facs else 1.0
    n = len(pairs)
    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.32 * n + 2)))
    c_ref, c_auto = SERIES[1], SERIES[0]
    for k, (E, qd, status, ref) in enumerate(pairs):
        off = 10 ** (decade_step * k)
        m = ref[:, 1] > 0
        o = np.argsort(ref[m, 0])
        ax.plot(ref[m, 0][o], ref[m, 1][o] * fac * off, "-", color=c_ref, lw=0.6,
                marker="o", ms=2.2, mew=0, alpha=0.9,
                label=f"{ref_label} × {fac:.3g}" if k == 0 else None)
        m = qd[:, 1] > 0
        ax.plot(qd[m, 0], qd[m, 1] * off, "-", color=c_auto, lw=1.1,
                label="auto (ReducedAuto)" if k == 0 else None)
        tag = "" if status in ("ok", "accepted") else f" [{status}]"
        ax.text(max(qd[m, 0].max(), ref[:, 0].max()) * 1.01, qd[m, 1][-1] * off,
                f"{E:.1f}{tag}", fontsize=6, va="center",
                color=INK if not tag else REVIEW)
    ax.set_yscale("log")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel(f"R × 10^(k·{decade_step:g})")
    ax.set_title(f"{sample}: auto vs {ref_label} (one factor {fac:.3g} on {ref_label} "
                 f"for the whole folder)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    return _save(fig, out, {"kind": "overview_waterfall_compare", "factor": fac,
                            "n": n, "per_energy_factor": facs}), fac


def overview_scales(plan, out, sample=""):
    """Scale factor per junction vs energy, flagged points marked; kept θ ranges."""
    ents = [pe for pe in plan["energies"] if pe.get("junctions_stitched")]
    pairs = sorted({s["pair"] for pe in ents for s in pe["junctions_stitched"]})
    fig, axes = plt.subplots(1 + len(pairs), 1, figsize=(9, 2.6 * (1 + len(pairs))),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for j, pair in enumerate(pairs):
        ax = axes[j]
        E, S, T, bad = [], [], [], []
        for pe in ents:
            for s in pe["junctions_stitched"]:
                if s["pair"] == pair:
                    E.append(pe["energy"])
                    S.append(s["scale"])
                    T.append(s.get("trend", np.nan))
                    bad.append(any(f["code"] in ("SCALE_OUTLIER", "SCALE_FROM_GAP",
                                                 "SCALE_FROM_TREND", "NO_OVERLAP")
                                   and pair in f["detail"]
                                   for f in pe.get("qc_flags", []) + pe["flags"]))
        E, S, T, bad = map(np.asarray, (E, S, T, bad))
        c = SERIES[(j + 1) % len(SERIES)]
        ax.plot(E, T, "-", color=INK2, lw=1, label="neighbour median")
        ax.plot(E[~bad], S[~bad], "o", color=c, ms=5, label=f"{pair} scale")
        if bad.any():
            ax.plot(E[bad], S[bad], "^", color=REVIEW, ms=8, mec=INK, mew=0.5,
                    label="flagged")
        ax.set_yscale("log")
        ax.set_ylabel("scale")
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    ax = axes[-1]
    for k, sid in enumerate([s["id"] for s in plan["sections"]]):
        E = [pe["energy"] for pe in ents for s in pe["sections"] if s["id"] == sid]
        lo = [s["theta0"] for pe in ents for s in pe["sections"] if s["id"] == sid]
        hi = [s["theta1"] for pe in ents for s in pe["sections"] if s["id"] == sid]
        c = SERIES[k % len(SERIES)]
        ax.vlines(E, lo, hi, color=c, lw=3, alpha=0.8, label=sid)
    ax.set_ylabel("kept θ (deg)")
    ax.set_xlabel("energy (eV)")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    _style(ax)
    fig.suptitle(f"{sample}: stitching scale factors and kept angular ranges",
                 fontsize=10, color=INK, x=0.02, ha="left")
    return _save(fig, out, {"kind": "overview_scales", "pairs": pairs})
