"""
NEXAFS peak fitting of absorption spectra or of the imaginary SLD from RSoXR.

Spectrum model (same math as NEXAFS_BuildingBlock / the INT_MOX notebooks):

    y(E) = baseline + slope·(E − slope_ref)
         + Σ_edges  NEXAFS.edge_bl_func(E, location, height, width, decay)
         + Σ_peaks  NEXAFS.gaussian(E, height, energy, width)      (width = FWHM)

Peak models are stored in project.json under "nexafs_models" and are built or
derived with ops, like layer models (`new_peak_model`, `derive_peak_model`).
Every parameter is {"value", "lo", "hi", "vary"}.  A fit applies a peak model
to one spectrum and writes <project>/nexafs/<fit name>/:

    spec.json    source, column, signal, energy window/shift, model snapshot
    result.json  fitted parameters (value, σ, bounds, at-bound flag), peak
                 areas, statistics (RMSE, R², χ²/ν, AIC, BIC)
    data.csv     E, y, σ, model, residual at the fitted points
    curve.csv    fine-grid total + every component (for plots / export)

Sources: 'Model:Layer' (fitted SLD in the project h5), a material name, or a
file.  Files: any delimited table; pick the y column by header name or index
(`column=`).  A 2-column file is treated as a NEXAFS intensity spectrum,
anything else as an SLD table whose Im column (index 2) is the default.

Signals for SLD sources: 'isld' (default; 1e-6 Å⁻², what the reference
notebooks fit), 'beta' (β), 'mu' (absorption coefficient, µm⁻¹ = 4πβ/λ — the
quantity NEXAFS measures; isld ∝ µ·E so the two differ by a slow tilt).
"""
import copy
import json
import os
import re

import numpy as np

HC = 12398.42                                   # eV·Å
GAUSS_AREA = float(np.sqrt(np.pi / (4 * np.log(2))))   # area = height·FWHM·this
PEAK_PARAMS = ("energy", "width", "height")
EDGE_PARAMS = ("location", "height", "width", "decay")
EDGE_DEFAULTS = {"C": 290.0, "N": 405.0, "O": 538.0, "F": 692.0, "Si": 1845.0}
SIGNALS = ("isld", "beta", "mu", "intensity")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def par(value, lo=None, hi=None, vary=None):
    """→ {"value", "lo", "hi", "vary"}.  par(v) is fixed; par(v, lo, hi) varies."""
    v = float(value)
    if lo is None and hi is None:
        return {"value": v, "lo": v, "hi": v, "vary": bool(vary) if vary is not None else False}
    lo = -np.inf if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    return {"value": v, "lo": lo, "hi": hi, "vary": True if vary is None else bool(vary)}


def _as_par(x):
    """Accept a number, (v, lo, hi[, vary]), or a par dict."""
    if isinstance(x, dict):
        d = {"value": float(x["value"]), "lo": float(x.get("lo", x["value"])),
             "hi": float(x.get("hi", x["value"])), "vary": bool(x.get("vary", False))}
        return d
    if isinstance(x, (list, tuple)):
        return par(*x)
    return par(x)


def _update_par(p, x):
    """Partial update: a number sets the value; a dict/tuple sets given fields."""
    p = dict(p)
    if isinstance(x, dict):
        for k in ("value", "lo", "hi"):
            if k in x and x[k] is not None:
                p[k] = float(x[k])
        if "vary" in x:
            p["vary"] = bool(x["vary"])
        if ("lo" in x or "hi" in x) and "vary" not in x:
            p["vary"] = True
    elif isinstance(x, (list, tuple)):
        p = par(*x)
    else:
        p["value"] = float(x)
    return p


def peak(name, energy, width, height, label=""):
    return {"name": name, "label": label, "energy": _as_par(energy),
            "width": _as_par(width), "height": _as_par(height)}


def edge(location, height, width=0.8, decay=0.01, name="edge"):
    return {"name": name, "location": _as_par(location), "height": _as_par(height),
            "width": _as_par(width), "decay": _as_par(decay)}


# ---------------------------------------------------------------------------
# peak models
# ---------------------------------------------------------------------------
def make_model(name, peaks, edges=(), baseline=0.0, slope=0.0, slope_ref=None,
               description="", parent=None, ops=None):
    peaks = [dict(p, energy=_as_par(p["energy"]), width=_as_par(p["width"]),
                  height=_as_par(p["height"]), label=p.get("label", ""))
             for p in peaks]
    edges = [dict(e, **{k: _as_par(e[k]) for k in EDGE_PARAMS}) for e in edges]
    if slope_ref is None:
        slope_ref = (edges[0]["location"]["value"] if edges else
                     float(np.mean([p["energy"]["value"] for p in peaks])) if peaks else 0.0)
    m = {"name": name, "description": description, "parent": parent, "ops": ops or [],
         "peaks": sorted(peaks, key=lambda p: p["energy"]["value"]), "edges": edges,
         "baseline": _as_par(baseline), "slope": _as_par(slope), "slope_ref": float(slope_ref)}
    validate_model(m)
    return m


def validate_model(m):
    names = [p["name"] for p in m["peaks"]] + [e["name"] for e in m["edges"]]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate peak/edge names in {names}")
    for key, p in _flat(m):
        if p["vary"] and not (p["lo"] <= p["value"] <= p["hi"]):
            raise ValueError(f"{m['name']}: {key} start {p['value']:g} outside "
                             f"[{p['lo']:g}, {p['hi']:g}]")
        if p["vary"] and p["lo"] >= p["hi"]:
            raise ValueError(f"{m['name']}: {key} varies but lo ≥ hi")
    for p in m["peaks"]:
        if p["width"]["value"] <= 0:
            raise ValueError(f"{p['name']}: width must be > 0")
    return True


def _flat(m):
    """Ordered [(key, par)] for every parameter of a model."""
    out = []
    for p in m["peaks"]:
        out += [(f"{p['name']}.{k}", p[k]) for k in PEAK_PARAMS]
    for e in m["edges"]:
        out += [(f"{e['name']}.{k}", e[k]) for k in EDGE_PARAMS]
    out += [("baseline", m["baseline"]), ("slope", m["slope"])]
    return out


def free_keys(m):
    return [k for k, p in _flat(m) if p["vary"]]


def _set_values(m, values):
    """Copy of model m with parameter values replaced from {key: value}."""
    m = copy.deepcopy(m)
    lookup = dict(_flat(m))
    for k, v in values.items():
        lookup[k]["value"] = float(v)
    return m


def evaluate(m, E, components=False):
    """Model spectrum at energies E (→ total, or (total, {component: array}))."""
    from NEXAFS import edge_bl_func, gaussian
    E = np.asarray(E, dtype=float)
    base = m["baseline"]["value"] + m["slope"]["value"] * (E - m["slope_ref"])
    comp = {"baseline": base}
    for e in m["edges"]:
        comp[e["name"]] = edge_bl_func(E, *(e[k]["value"] for k in EDGE_PARAMS))
    for p in m["peaks"]:
        comp[p["name"]] = gaussian(E, p["height"]["value"], p["energy"]["value"],
                                   p["width"]["value"])
    total = np.sum(list(comp.values()), axis=0)
    return (total, comp) if components else total


def peak_area(p):
    return p["height"]["value"] * p["width"]["value"] * GAUSS_AREA


def model_table(m):
    """Plain-text table of a peak model (bounds; '·' = fixed)."""
    def fmt(p):
        if not p["vary"]:
            return f"{p['value']:8.3f}  ·"
        return f"{p['value']:8.3f} [{p['lo']:g}, {p['hi']:g}]"
    lines = [f"peak model {m['name']}" + (f" (from {m['parent']})" if m.get("parent") else "")
             + (f": {m['description']}" if m.get("description") else ""),
             f"{'peak':10s} {'energy (eV)':26s} {'FWHM (eV)':24s} {'height':24s} label"]
    for p in m["peaks"]:
        lines.append(f"{p['name']:10s} {fmt(p['energy']):26s} {fmt(p['width']):24s} "
                     f"{fmt(p['height']):24s} {p.get('label', '')}")
    for e in m["edges"]:
        lines.append(f"{e['name']:10s} location {fmt(e['location'])}  height {fmt(e['height'])}"
                     f"  width {fmt(e['width'])}  decay {fmt(e['decay'])}")
    lines.append(f"baseline {fmt(m['baseline'])}   slope {fmt(m['slope'])} "
                 f"(about {m['slope_ref']:g} eV)")
    lines.append(f"{len(free_keys(m))} free parameters")
    return "\n".join(lines)


def from_notebook_dicts(name, peak_params, edge_params=None, baseline=0.0, **kw):
    """
    Convert the INT_MOX notebook format ({"peak_1_energy": v,
    "peak_1_energy_bounds": (lo, hi, vary), ..., "peak_1_color": c} and an
    edge dict with optional "<param>_bounds") into a peak model.  Parameters
    without bounds are fixed.  A start outside its (lo, hi) window — common in
    those notebooks for fixed parameters — is kept for fixed parameters and
    clipped into the window for varied ones.
    """
    def conv(d, key):
        v = d[key]
        b = d.get(f"{key}_bounds")
        if b is None:
            return par(v)
        lo, hi = float(b[0]), float(b[1])
        vary = bool(b[2]) if len(b) > 2 else True
        if not vary:
            return {"value": float(v), "lo": lo, "hi": hi, "vary": False}
        return {"value": float(np.clip(v, lo, hi)), "lo": lo, "hi": hi, "vary": True}

    names = sorted({m.group(1) for k in peak_params
                    for m in [re.match(r"(peak_\d+)_(energy|width|height)$", k)] if m},
                   key=lambda s: int(s.split("_")[1]))
    peaks = [{"name": n.replace("peak_", "p"), "label": "",
              **{k: conv(peak_params, f"{n}_{k}") for k in PEAK_PARAMS}} for n in names]
    edges = []
    if edge_params:
        edges = [{"name": "edge", **{k: conv(edge_params, k) for k in EDGE_PARAMS}}]
    return make_model(name, peaks, edges, baseline=baseline, **kw)


# ----------------------------- derive ops ----------------------------------
def _find(m, name):
    for coll in ("peaks", "edges"):
        for x in m[coll]:
            if x["name"] == name:
                return coll, x
    raise KeyError(f"no peak/edge '{name}' in {m['name']}; have "
                   f"{[p['name'] for p in m['peaks'] + m['edges']]}")


def _targets(m, which, coll="peaks"):
    if which in (None, "all"):
        return m[coll]
    which = [which] if isinstance(which, str) else which
    return [_find(m, n)[1] for n in which]


def _next_name(m):
    used = {int(p["name"][1:]) for p in m["peaks"] if re.fullmatch(r"p\d+", p["name"])}
    return f"p{max(used, default=0) + 1}"


def apply_ops(m, ops, fits_lookup=None):
    """
    Ops (list of dicts) on a peak model; returns a new model.
      {"op": "add_peak", "name"?, "energy", "width", "height", "label"?}
      {"op": "remove_peak", "name"}
      {"op": "set_peak", "name", "energy"|"width"|"height": number | (v, lo, hi[, vary]) | {...}, "label"?}
      {"op": "set_edge", "name"?="edge", <edge param>: ...}   {"op": "add_edge", ...}  {"op": "remove_edge", "name"}
      {"op": "set_baseline", "baseline"?: ..., "slope"?: ..., "slope_ref"?: E}
      {"op": "freeze" | "free", "params": ["energy", "width", ...], "peaks"?: [...] | "all",
       "edges"?: bool}        — vary False/True (free needs finite bounds)
      {"op": "window", "param": "energy", "pm": 0.3, "peaks"?: ...} — bounds = value ± pm (varied)
      {"op": "shift_energies", "delta": dE}   — peaks + edges, values and bounds
      {"op": "scale_heights", "factor": f}    — peak/edge heights, baseline, slope
      {"op": "seed_from_fit", "fit": NAME}    — start values from a fit (clipped to bounds)
    """
    m = copy.deepcopy(m)
    for op in ops:
        kind = op["op"]
        if kind == "add_peak":
            m["peaks"].append(peak(op.get("name") or _next_name(m), op["energy"], op["width"],
                                   op["height"], op.get("label", "")))
        elif kind == "remove_peak":
            _find(m, op["name"])                       # raises if unknown
            m["peaks"] = [p for p in m["peaks"] if p["name"] != op["name"]]
        elif kind in ("set_peak", "set_edge"):
            nm = op.get("name", "edge" if kind == "set_edge" else None)
            _, x = _find(m, nm)
            for k in (PEAK_PARAMS if kind == "set_peak" else EDGE_PARAMS):
                if k in op:
                    x[k] = _update_par(x[k], op[k])
            if "label" in op:
                x["label"] = op["label"]
            if "rename" in op:
                x["name"] = op["rename"]
        elif kind == "add_edge":
            m["edges"].append(edge(op["location"], op["height"], op.get("width", 0.8),
                                   op.get("decay", 0.01), name=op.get("name", f"edge{len(m['edges']) + 1}")))
        elif kind == "remove_edge":
            m["edges"] = [e for e in m["edges"] if e["name"] != op["name"]]
        elif kind == "set_baseline":
            for k in ("baseline", "slope"):
                if k in op:
                    m[k] = _update_par(m[k], op[k])
            if "slope_ref" in op:
                m["slope_ref"] = float(op["slope_ref"])
        elif kind in ("freeze", "free"):
            params = op.get("params", list(PEAK_PARAMS))
            params = [params] if isinstance(params, str) else params
            objs = _targets(m, op.get("peaks", "all"))
            if op.get("edges"):
                objs = objs + m["edges"]
            for x in objs:
                for k in params:
                    if k in x:
                        if kind == "free" and not np.isfinite([x[k]["lo"], x[k]["hi"]]).all():
                            raise ValueError(f"free {x['name']}.{k}: give finite bounds first")
                        if kind == "free" and x[k]["lo"] == x[k]["hi"]:
                            raise ValueError(f"free {x['name']}.{k}: bounds are a point; use window/set_peak")
                        x[k]["vary"] = kind == "free"
            for k in ("baseline", "slope"):
                if k in params:
                    m[k]["vary"] = kind == "free"
        elif kind == "window":
            k, pm = op["param"], float(op["pm"])
            objs = _targets(m, op.get("peaks", "all"))
            if op.get("edges"):
                objs = objs + m["edges"]
            for x in objs:
                if k in x:
                    v = x[k]["value"]
                    lo = v - pm if not op.get("relative") else v * (1 - pm)
                    hi = v + pm if not op.get("relative") else v * (1 + pm)
                    if k in ("width", "height", "decay"):
                        lo = max(lo, 0.0 if k != "width" else 1e-3)
                    x[k] = {"value": v, "lo": lo, "hi": hi, "vary": True}
        elif kind == "shift_energies":
            d = float(op["delta"])
            for x in [p["energy"] for p in m["peaks"]] + [e["location"] for e in m["edges"]]:
                for f in ("value", "lo", "hi"):
                    x[f] += d
            m["slope_ref"] += d
        elif kind == "scale_heights":
            f = float(op["factor"])
            for x in ([p["height"] for p in m["peaks"]] + [e["height"] for e in m["edges"]]
                      + [m["baseline"], m["slope"]]):
                for k in ("value", "lo", "hi"):
                    x[k] *= f
                if f < 0:
                    x["lo"], x["hi"] = x["hi"], x["lo"]
        elif kind == "seed_from_fit":
            if fits_lookup is None:
                raise ValueError("seed_from_fit needs a project")
            vals = fits_lookup(op["fit"])
            lookup = dict(_flat(m))
            for k, v in vals.items():
                if k in lookup:
                    p = lookup[k]
                    p["value"] = float(np.clip(v, p["lo"], p["hi"])) if p["vary"] else float(v)
        else:
            raise ValueError(f"unknown op {kind!r}")
    m["peaks"] = sorted(m["peaks"], key=lambda p: p["energy"]["value"])
    validate_model(m)
    return m


# ---------------------------------------------------------------------------
# project storage
# ---------------------------------------------------------------------------
def _models(project):
    return project.cfg.setdefault("nexafs_models", {})


def model_names(project):
    return sorted(project.cfg.get("nexafs_models", {}))


def get_model(project, name):
    try:
        return copy.deepcopy(project.cfg["nexafs_models"][name])
    except KeyError:
        raise KeyError(f"no peak model '{name}'; known: {model_names(project)}") from None


def save_model(project, m, overwrite=False):
    cur = _models(project)
    if m["name"] in cur and not overwrite:
        if cur[m["name"]] == m:
            return m
        raise ValueError(f"peak model '{m['name']}' exists (overwrite=True to replace)")
    validate_model(m)
    cur[m["name"]] = json.loads(json.dumps(m, default=float))
    project.save()
    return m


def new_peak_model(project, name, peaks, edges=(), overwrite=False, **kw):
    return save_model(project, make_model(name, peaks, edges, **kw), overwrite)


def derive_peak_model(project, parent, name, ops, description="", overwrite=False):
    m = apply_ops(get_model(project, parent), ops,
                  fits_lookup=lambda f: load_fit(project, f)["values"])
    m.update(name=name, parent=parent, description=description or m.get("description", ""),
             ops=list(ops))
    return save_model(project, m, overwrite)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _read_table(path):
    """→ (ndarray, column names or None)."""
    import pandas as pd
    with open(path) as fh:
        first = fh.readline()
    sep = "," if "," in first else r"\s+"
    try:
        [float(x) for x in re.split(r"[,\s]+", first.strip())]
        header = None
    except ValueError:
        header = 0
    df = pd.read_csv(path, sep=sep, header=header, engine="python", comment="#")
    names = [str(c).strip() for c in df.columns] if header == 0 else None
    return df.to_numpy(dtype=float), names


def _col(names, c, ncol):
    if c is None:
        return None
    if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.lstrip("-").isdigit()):
        i = int(c)
        if not -ncol <= i < ncol:
            raise IndexError(f"column {i} out of range ({ncol} columns)")
        return i % ncol
    if names is None or c not in names:
        raise KeyError(f"column '{c}' not found; columns: {names}")
    return names.index(c)


def load_signal(project, source, column=None, signal="auto", sigma_column=None,
                energy_shift=0.0, criteria="best"):
    """
    → dict(E, y, sigma|None, signal, provenance, column).
    signal 'auto': 'isld' for SLD sources, 'intensity' for 2-column spectra.
    The energy shift is applied before any isld → β/µ conversion.
    """
    sig = None
    if os.path.exists(source):
        arr, names = _read_table(source)
        ncol = arr.shape[1]
        ci = _col(names, column, ncol)
        native = "intensity" if ncol == 2 else "isld"
        if ci is None:
            ci = 1 if ncol == 2 else 2
        if ci == 1 and ncol > 2 and column is not None and native == "isld":
            native = "sld"                      # explicitly asked for the real part
        if names and column is not None:
            n = names[ci].lower()
            native = ("isld" if ("isld" in n or "imag" in n) else
                      "sld" if ("sld" in n or "real" in n) else
                      "intensity" if ncol == 2 else native)
        si = _col(names, sigma_column, ncol)
        if si is None and sigma_column is None and ncol == 5 and ci == 2 and (
                names is None or "std" in names[4].lower()):
            si = 4                               # sld_from_jaxns_cache layout
        E, y = arr[:, 0], arr[:, ci]
        sig = arr[:, si] if si is not None else None
        prov = f"file {source} [{names[ci] if names else f'col {ci}'}]"
        col_label = names[ci] if names else ci
    else:
        from .stoich import resolve_source
        arr, prov = resolve_source(project, source, criteria)
        ci = 2 if column is None else int(column)
        native = "isld" if ci == 2 else "sld"
        E, y = arr[:, 0], arr[:, ci]
        col_label = ci
    o = np.argsort(E)
    E, y = E[o] + float(energy_shift), y[o]
    sig = sig[o] if sig is not None else None
    good = np.isfinite(E) & np.isfinite(y)
    E, y = E[good], y[good]
    sig = sig[good] if sig is not None else None

    signal = native if signal in (None, "auto") else signal
    if signal not in SIGNALS + ("sld",):
        raise ValueError(f"signal must be one of {SIGNALS}")
    if signal != native:
        if native != "isld" or signal not in ("beta", "mu"):
            raise ValueError(f"cannot convert {native} → {signal} (only isld → beta | mu)")
        lam = HC / E
        f = 1e-6 * lam ** 2 / (2 * np.pi)                     # isld → β
        if signal == "mu":
            f = f * 4 * np.pi / lam * 1e4                     # β → µ (µm⁻¹)
        y = y * f
        sig = sig * f if sig is not None else None
    return {"E": E, "y": y, "sigma": sig, "signal": signal, "provenance": prov,
            "column": col_label}


SIGNAL_UNITS = {"isld": "Im SLD (10⁻⁶ Å⁻²)", "sld": "Re SLD (10⁻⁶ Å⁻²)", "beta": "β",
                "mu": "µ (µm⁻¹)", "intensity": "intensity (a.u.)"}


# ---------------------------------------------------------------------------
# starting-model suggestion
# ---------------------------------------------------------------------------
def suggest_model(E, y, name="suggested", element="C", erange=None, edge_location=None,
                  scales=(0.15, 1.2), rel_prominence=0.04, min_sep=0.4, max_peaks=14,
                  step=0.05):
    """
    Starting peak model from the data: peaks at minima of the smoothed second
    derivative (curvature), at a narrow scale (sharp π* / Rydberg features) and
    a broad scale (σ* above the edge); heights from the data minus a rough
    baseline + edge; FWHM from the curvature (Gaussian: y'' = −h/σ²).  One step
    edge at the element's ionisation potential (C 290, N 405, O 538 eV) unless
    edge_location is given.  A proposal to refine, not a fit.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    E, y = np.asarray(E, float), np.asarray(y, float)
    if erange:
        k = (E >= erange[0]) & (E <= erange[1])
        E, y = E[k], y[k]
    g = np.arange(E.min(), E.max() + step / 2, step)
    yg = np.interp(g, E, y)
    dE = float(np.median(np.diff(E)))
    scales = sorted(max(s, 1.5 * dE) for s in scales)   # never finer than the data
    loc = float(edge_location if edge_location is not None else EDGE_DEFAULTS.get(element, g.mean()))
    pre = g < loc - 6
    base = float(np.median(yg[pre])) if pre.sum() > 3 else float(np.percentile(yg, 5))
    post = (g > loc + 25) & (g < loc + 45)
    decay0 = 0.01
    eh = (float(np.median(yg[post])) - base) / np.exp(-decay0 * 35) if post.sum() > 3 else \
        float(np.percentile(yg, 90) - base) / 2
    eh = max(eh, 1e-6)
    from NEXAFS import edge_bl_func
    bg = base + edge_bl_func(g, loc, eh, 0.8, decay0)

    cands = []
    for s in scales:
        ys = gaussian_filter1d(yg, max(s / step, 1.0))
        d2 = np.gradient(np.gradient(ys, step), step)
        c = -d2
        if s > min(scales):
            c = np.where(g > loc - 1, c, 0)        # broad scale only for σ* region
        if c.max() <= 0:
            continue
        idx, pr = find_peaks(c, prominence=rel_prominence * c.max(),
                             distance=max(1, int(min_sep / step)))
        for i, prom in zip(idx, pr["prominences"]):
            h = float(ys[i] - bg[i])
            if h <= 0:
                continue
            sigma = np.sqrt(h / max(c[i], 1e-12))
            w = float(np.clip(2.3548 * sigma, *((0.3, 3.0) if s == min(scales) else (1.5, 12.0))))
            cands.append({"energy": float(g[i]), "height": h, "width": w,
                          "score": prom / c.max(), "scale": s})
    cands.sort(key=lambda d: -d["score"])
    chosen = []
    for d in cands:
        if all(abs(d["energy"] - c["energy"]) >= max(min_sep, 0.35 * min(d["width"], c["width"]))
               for c in chosen):
            chosen.append(d)
        if len(chosen) >= max_peaks:
            break
    chosen.sort(key=lambda d: d["energy"])
    peaks = []
    for i, d in enumerate(chosen, 1):
        narrow = d["scale"] == min(scales)
        de = 0.3 if narrow else 1.5
        w = d["width"]
        peaks.append(peak(f"p{i}", (d["energy"], d["energy"] - de, d["energy"] + de),
                          (w, max(0.2, 0.5 * w), min(15.0, 2.0 * w)),
                          (d["height"], 0.0, 3.0 * d["height"])))
    edges = [edge((loc, loc - 1.5, loc + 1.5), (eh, 0.0, 3 * eh), (0.8, 0.3, 3.0),
                  (decay0, 0.0, 0.1))]
    return make_model(name, peaks, edges, baseline=(base, base - abs(base) - 1e-6,
                                                    base + abs(base) + 1e-6),
                      description=f"suggested from data ({element} edge at {loc:g} eV)")


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fits_dir(project, name=None):
    d = project.dir("nexafs")
    return os.path.join(d, name) if name else d


def default_fit_name(model, source, column=None):
    tag = os.path.splitext(os.path.basename(source))[0] if os.path.exists(source) else source
    if column is not None and not isinstance(column, int):
        tag += f"_{column}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model}__{tag}").strip("_")


def fit_spectrum(m, E, y, sigma=None, n_starts=0, seed=0, max_nfev=20000):
    """
    Bounded least squares (scipy trf, like curve_fit(method='trf')) of model m
    to (E, y); optional random multi-start inside the bounds.
    → (fitted model, info dict).
    """
    from scipy.optimize import least_squares
    keys = free_keys(m)
    if not keys:
        raise ValueError("no free parameters")
    if len(E) <= len(keys):
        raise ValueError(f"{len(E)} points for {len(keys)} free parameters")
    lookup = dict(_flat(m))
    x0 = np.array([lookup[k]["value"] for k in keys])
    lo = np.array([lookup[k]["lo"] for k in keys])
    hi = np.array([lookup[k]["hi"] for k in keys])
    w = 1.0 / np.asarray(sigma) if sigma is not None else np.ones_like(y)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("sigma must be finite and > 0")
    work = copy.deepcopy(m)
    wl = dict(_flat(work))

    def resid(x):
        for k, v in zip(keys, x):
            wl[k]["value"] = v
        return (evaluate(work, E) - y) * w

    def run(xs):
        return least_squares(resid, xs, bounds=(lo, hi), method="trf", x_scale="jac",
                             max_nfev=max_nfev)

    best = run(x0)
    starts = [best.cost]
    rng = np.random.default_rng(seed)
    finite = np.isfinite(lo) & np.isfinite(hi)
    for _ in range(int(n_starts)):
        xs = x0.copy()
        xs[finite] = lo[finite] + rng.random(finite.sum()) * (hi[finite] - lo[finite])
        r = run(xs)
        starts.append(r.cost)
        if r.cost < best.cost:
            best = r
    x = best.x
    r = resid(x)
    n, k = len(y), len(keys)
    dof = n - k
    chi2 = float(np.sum(r ** 2))
    J = best.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    s2 = chi2 / dof if dof > 0 else np.nan
    if sigma is None:
        cov = cov * s2                       # σ estimated from the residuals
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    fitted = _set_values(m, dict(zip(keys, x)))
    yfit = evaluate(fitted, E)
    res = y - yfit
    ssr = float(np.sum(res ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    if sigma is None:
        aic = n * np.log(ssr / n) + 2 * k
        bic = n * np.log(ssr / n) + k * np.log(n)
    else:
        aic, bic = chi2 + 2 * k, chi2 + k * np.log(n)
    span = hi - lo
    at_lo = finite & ((x - lo) <= 0.01 * span)
    at_hi = finite & ((hi - x) <= 0.01 * span)
    info = {"keys": keys, "x": x.tolist(), "stderr": err.tolist(),
            "corr": (cov / np.outer(err, err)).tolist() if np.all(err > 0) else None,
            "at_bound": {kk: ("lo" if a else "hi") for kk, a, b in zip(keys, at_lo, at_hi) if a or b},
            "n_points": n, "n_free": k, "dof": dof, "weighted": sigma is not None,
            "rmse": float(np.sqrt(ssr / n)), "r_squared": 1 - ssr / sst if sst > 0 else np.nan,
            "chi2": chi2, "chi2_red": s2, "aic": float(aic), "bic": float(bic),
            "success": bool(best.success), "message": best.message, "nfev": int(best.nfev),
            "start_costs": sorted(float(c) for c in starts)}
    return fitted, info


def run_fit(project, model, source, name=None, column=None, signal="auto",
            erange=None, energy_shift=0.0, sigma_column=None, weights="auto",
            n_starts=0, seed=0, criteria="best", overwrite=True):
    """
    Fit a stored peak model to a spectrum and write <project>/nexafs/<name>/.
    weights: 'auto' (σ if the source has it), 'none', 'sigma'.
    → summary dict (result.json contents).
    """
    import pandas as pd
    m = get_model(project, model) if isinstance(model, str) else model
    d = load_signal(project, source, column, signal, sigma_column, energy_shift, criteria)
    E, y, sig = d["E"], d["y"], d["sigma"]
    if erange:
        k = (E >= erange[0] - 1e-9) & (E <= erange[1] + 1e-9)
        E, y = E[k], y[k]
        sig = sig[k] if sig is not None else None
    if weights == "none":
        sig = None
    elif weights == "sigma" and sig is None:
        raise ValueError("weights='sigma' but the source has no σ column")
    name = name or default_fit_name(m["name"], source, column)
    out = fits_dir(project, name)
    if os.path.exists(os.path.join(out, "result.json")) and not overwrite:
        raise ValueError(f"fit '{name}' exists")
    fitted, info = fit_spectrum(m, E, y, sig, n_starts=n_starts, seed=seed)

    lookup = dict(_flat(fitted))
    errs = dict(zip(info["keys"], info["stderr"]))
    params = []
    for key, p in _flat(fitted):
        params.append({"param": key, "value": p["value"], "stderr": errs.get(key),
                       "lo": p["lo"], "hi": p["hi"], "vary": p["vary"],
                       "at_bound": info["at_bound"].get(key, "")})
    eh = sum(e["height"]["value"] for e in fitted["edges"]) or np.nan
    peaks = []
    for p in fitted["peaks"]:
        a = peak_area(p)
        kh, kw = f"{p['name']}.height", f"{p['name']}.width"
        rel = np.hypot((errs.get(kh) or 0) / p["height"]["value"] if p["height"]["value"] else 0,
                       (errs.get(kw) or 0) / p["width"]["value"])
        peaks.append({"name": p["name"], "label": p.get("label", ""),
                      "energy": p["energy"]["value"], "fwhm": p["width"]["value"],
                      "height": p["height"]["value"], "area": a, "area_stderr": abs(a) * rel,
                      "area_over_edge": a / eh})
    tot = sum(pk["area"] for pk in peaks) or np.nan
    for pk in peaks:
        pk["area_fraction"] = pk["area"] / tot
    summary = {"name": name, "model": m["name"], "source": source,
               "provenance": d["provenance"], "column": d["column"], "signal": d["signal"],
               "erange": list(erange) if erange else [float(E.min()), float(E.max())],
               "energy_shift": float(energy_shift), "values": dict(zip(info["keys"], info["x"])),
               "all_values": {k: p["value"] for k, p in lookup.items()},
               "params": params, "peaks": peaks,
               "edge_height_total": eh,
               **{k: info[k] for k in ("n_points", "n_free", "dof", "weighted", "rmse",
                                       "r_squared", "chi2", "chi2_red", "aic", "bic",
                                       "success", "message", "nfev", "at_bound",
                                       "start_costs")},
               "n_starts": int(n_starts)}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "spec.json"), "w") as fh:
        json.dump({"model": m, "source": source, "column": column, "signal": signal,
                   "erange": erange, "energy_shift": energy_shift, "sigma_column": sigma_column,
                   "weights": weights, "n_starts": n_starts, "seed": seed,
                   "criteria": criteria}, fh, indent=2, default=float)
    with open(os.path.join(out, "fitted_model.json"), "w") as fh:
        json.dump(fitted, fh, indent=2, default=float)
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    yfit = evaluate(fitted, E)
    pd.DataFrame({"energy_eV": E, "y": y, "sigma": sig if sig is not None else np.nan,
                  "model": yfit, "residual": y - yfit}).to_csv(
        os.path.join(out, "data.csv"), index=False)
    g = np.arange(E.min(), E.max() + 0.025, 0.05)
    tot_g, comp = evaluate(fitted, g, components=True)
    pd.DataFrame({"energy_eV": g, "total": tot_g, **comp}).to_csv(
        os.path.join(out, "curve.csv"), index=False)
    return summary


def fit_names(project):
    d = fits_dir(project)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.exists(os.path.join(d, n, "result.json")))


def load_fit(project, name):
    """→ result.json dict plus 'fitted_model', 'data' and 'curve' DataFrames."""
    import pandas as pd
    d = fits_dir(project, name)
    if not os.path.exists(os.path.join(d, "result.json")):
        raise KeyError(f"no NEXAFS fit '{name}'; known: {fit_names(project)}")
    with open(os.path.join(d, "result.json")) as fh:
        s = json.load(fh)
    with open(os.path.join(d, "fitted_model.json")) as fh:
        s["fitted_model"] = json.load(fh)
    s["data"] = pd.read_csv(os.path.join(d, "data.csv"))
    s["curve"] = pd.read_csv(os.path.join(d, "curve.csv"))
    return s


def format_fit(s, params=True):
    lines = [f"NEXAFS fit '{s['name']}': model {s['model']} → {s['provenance']}",
             f"  signal {s['signal']}, {s['erange'][0]:g}–{s['erange'][1]:g} eV"
             + (f", energy shift {s['energy_shift']:+g} eV" if s["energy_shift"] else "")
             + f"; {s['n_points']} points, {s['n_free']} free"
             + (", σ-weighted" if s["weighted"] else "")
             + (f", best of {s['n_starts'] + 1} starts" if s.get("n_starts") else ""),
             f"  RMSE {s['rmse']:.4g}  R² {s['r_squared']:.4f}  χ²/ν {s['chi2_red']:.4g}  "
             f"AIC {s['aic']:.1f}  BIC {s['bic']:.1f}"
             + ("" if s["success"] else f"  (optimizer: {s['message']})")]
    if params:
        lines.append(f"  {'peak':8s} {'E (eV)':>8s} {'FWHM':>6s} {'height':>8s} {'area':>8s} "
                     f"{'±':>6s} {'frac':>5s}  label")
        for pk in s["peaks"]:
            lines.append(f"  {pk['name']:8s} {pk['energy']:8.2f} {pk['fwhm']:6.2f} "
                         f"{pk['height']:8.3f} {pk['area']:8.3f} {pk['area_stderr']:6.3f} "
                         f"{pk['area_fraction']:5.2f}  {pk['label']}")
        for p in s["params"]:
            if p["param"].split(".")[0] in [e["name"] for e in s["fitted_model"]["edges"]] \
                    or p["param"] in ("baseline", "slope"):
                if p["vary"] or p["param"] == "baseline":
                    e = f" ± {p['stderr']:.3g}" if p["stderr"] else ""
                    lines.append(f"  {p['param']:16s} {p['value']:.4g}{e}"
                                 + ("" if p["vary"] else " (fixed)"))
    if s["at_bound"]:
        lines.append("  AT BOUND: " + ", ".join(f"{k} ({v})" for k, v in s["at_bound"].items()))
    return "\n".join(lines)


def peak_series(project, names, peak=None, ratio=None, normalize_by_edge=False, by="name"):
    """
    Peak area (or area ratio) across fits.  peak/ratio refer to peak names
    (by='name', robust when every fit derives from one model) or to the
    nearest fitted energy (by='energy', peak = float eV).
    → pandas DataFrame [fit, value, stderr, energy(s)].
    """
    import pandas as pd

    def get(s, ref):
        pks = s["peaks"]
        if by == "energy":
            pk = min(pks, key=lambda p: abs(p["energy"] - float(ref)))
        else:
            pk = next((p for p in pks if p["name"] == ref), None)
            if pk is None:
                raise KeyError(f"fit {s['name']}: no peak '{ref}'")
        a, e = pk["area"], pk["area_stderr"]
        if normalize_by_edge:
            a, e = a / s["edge_height_total"], e / s["edge_height_total"]
        return a, e, pk["energy"]

    rows = []
    for n in names:
        s = load_fit(project, n)
        if ratio:
            (a, ea, Ea), (b, eb, Eb) = get(s, ratio[0]), get(s, ratio[1])
            v = a / b
            rows.append({"fit": n, "value": v, "stderr": abs(v) * np.hypot(ea / a, eb / b),
                         "energy": Ea, "energy_b": Eb})
        else:
            a, ea, Ea = get(s, peak)
            rows.append({"fit": n, "value": a, "stderr": ea, "energy": Ea})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# export to optical constants (KK)
# ---------------------------------------------------------------------------
def export_material(project, fit, formula, density, register_as=None, step=0.1, out=None):
    """
    Turn a fitted isld spectrum into a KK-consistent optical-constant table:
    Im = peak model on a `step` grid over the fit window (spliced into the
    formula's tabulated scattering factors outside it, kkcalc), Re = its KK
    transform.  Writes CSV Energy_eV,Real_SLD,Imag_SLD; optionally registers it
    as a project material (starting optical constants for RSoXR fits).
    kkcalc rescales the spliced Im affinely onto the formula's tabulated tails,
    so the table's Im differs from the peak model by `im_scale` (reported, and
    written to <fit>/kk_export.json); the exported Re/Im pair is the
    self-consistent one.  → (csv path, info dict).
    """
    from .stoich import kk_sld
    s = load_fit(project, fit)
    if s["signal"] != "isld":
        raise ValueError(f"fit '{fit}' is on signal {s['signal']}; export needs an isld fit")
    lo, hi = s["erange"]
    g = np.arange(lo, hi + step / 2, step)
    im = evaluate(s["fitted_model"], g)
    sld = np.column_stack([g, np.zeros_like(g), im])
    kk = kk_sld(sld, formula, float(density), [float(lo), float(hi)])
    out = out or os.path.join(fits_dir(project, fit), f"kk_{formula}_{density:g}.csv")
    np.savetxt(out, kk, delimiter=",", header="Energy_eV,Real_SLD,Imag_SLD", comments="")
    im_kk = np.interp(g, kk[:, 0], kk[:, 2])
    big = im > 0.2 * im.max()
    info = {"fit": fit, "formula": formula, "density": float(density), "csv": out,
            "erange": [float(lo), float(hi)],
            "im_scale": float(np.median(im_kk[big] / im[big])),
            "registered_as": register_as}
    with open(os.path.join(fits_dir(project, fit), "kk_export.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    if register_as:
        project.add_material(register_as, kind="file", path=out)
    return out, info
