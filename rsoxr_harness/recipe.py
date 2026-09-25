"""
recipe.py
---------
Declarative, JSON-serialisable model recipes and the derivation ops used to
build "Model N+1 = Model N + change".

A ModelRecipe is everything generate_batch_models needs except the data and
the tabulated SLD arrays:

    layers      – top → bottom (this list IS layer_order)
    instrument  – scale / bkg / dq settings

Each LayerSpec names the *material* whose tabulated SLD seeds it, so several
layers can share one source (e.g. SOG2 → SOG) without duplicating arrays.
"""

import copy
import hashlib
import json

import numpy as np
from dataclasses import dataclass, field, asdict, fields


# ---------------------------------------------------------------------------
# Bound helpers
# ---------------------------------------------------------------------------

def _bound(b):
    """Normalise a (lo, hi, vary) bound from JSON lists / tuples."""
    if b is None:
        return None
    lo, hi, vary = b
    return (float(lo), float(hi), bool(vary))


def _offset(d):
    """Normalise an sld_offset dict {'real': (lo, hi, vary), 'imag': ...}."""
    if d is None:
        return None
    return {k: _bound(v) for k, v in d.items() if v is not None}


def pm(value, vary=True):
    """'±value' → (-value, value, vary)."""
    return (-float(value), float(value), bool(vary))


def pm_offset(real, imag=None, vary=True):
    """'SLD ±real (imag ±imag)' → sld_offset dict.  imag defaults to real."""
    imag = real if imag is None else imag
    return {"real": pm(real, vary), "imag": pm(imag, vary)}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

_BOUND_FIELDS = ("thickness_bounds", "roughness_bounds",
                 "sld_real_bounds", "sld_imag_bounds")


@dataclass
class LayerSpec:
    """
    One slab.  material=None means an energy-independent constant SLD
    (constant_sld, default 0 + 0j – i.e. air).

    sld_offset       : bounds relative to the tabulated SLD at each energy
    sld_real_bounds /
    sld_imag_bounds  : absolute bounds; override sld_offset when given
    Unbounded SLDs are fixed at the tabulated value.
    """
    name: str
    material: str | None = None
    thickness: float = 0.0
    roughness: float = 0.0
    thickness_bounds: tuple | None = None
    roughness_bounds: tuple | None = None
    sld_offset: dict | None = None
    sld_real_bounds: tuple | None = None
    sld_imag_bounds: tuple | None = None
    constant_sld: dict | None = None

    def __post_init__(self):
        self.thickness = float(self.thickness)
        self.roughness = float(self.roughness)
        for f in _BOUND_FIELDS:
            setattr(self, f, _bound(getattr(self, f)))
        self.sld_offset = _offset(self.sld_offset)
        if self.material is None and self.constant_sld is None:
            self.constant_sld = {"real": 0.0, "imag": 0.0}

    @property
    def is_constant(self):
        return self.material is None

    def to_json(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def layer_params(self):
        """Entry for generate_batch_models' base_layer_params."""
        p = {"thickness": self.thickness, "roughness": self.roughness}
        for f in _BOUND_FIELDS:
            v = getattr(self, f)
            if v is not None:
                p[f] = v
        if self.is_constant:
            p.setdefault("thickness_bounds", (self.thickness, self.thickness, False))
            p.setdefault("roughness_bounds", (self.roughness, self.roughness, False))
            p.setdefault("sld_real_bounds", (self.constant_sld["real"],) * 2 + (False,))
            p.setdefault("sld_imag_bounds", (self.constant_sld["imag"],) * 2 + (False,))
        return p


@dataclass
class InstrumentSpec:
    """scale_bounds / bkg_bounds are multiplicative factors (Model_Setup
    convention); dq_bounds are absolute.  bkg=None → data minimum."""
    scale: float = 1.0
    scale_bounds: tuple = (0.1, 10.0)
    vary_scale: bool = True
    bkg: float | None = None
    bkg_bounds: tuple = (0.01, 10.0)
    vary_bkg: bool = True
    dq: float = 1.6
    dq_bounds: tuple = (1.0, 2.0)
    vary_dq: bool = False

    def __post_init__(self):
        for f in ("scale_bounds", "bkg_bounds", "dq_bounds"):
            setattr(self, f, tuple(float(x) for x in getattr(self, f)))

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def generate_kwargs(self):
        return asdict(self)


@dataclass
class ModelRecipe:
    name: str
    layers: list
    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)
    parent: str | None = None
    ops: list = field(default_factory=list)
    description: str = ""
    overrides: list = field(default_factory=list)
    seed: dict | None = None      # {"model", "criteria", "runs": {E: run}}
    links: dict | None = None     # {"params": {layer: [thickness|roughness]},
                                  #  "energies": [E, ...] | None (= all data)}

    def __post_init__(self):
        self.layers = [l if isinstance(l, LayerSpec) else LayerSpec.from_json(l)
                       for l in self.layers]
        if isinstance(self.instrument, dict):
            self.instrument = InstrumentSpec.from_json(self.instrument)
        self.overrides = sorted((_norm_override(o) for o in self.overrides),
                                key=lambda o: (o["energy"], o["layer"]))
        self.links = _norm_links(self.links)

    @property
    def is_linked(self):
        """True if some parameter is shared across energies (global fit)."""
        return bool(self.links and self.links["params"])

    def overrides_at(self, energy):
        return [o for o in self.overrides if abs(o["energy"] - float(energy)) < 1e-3]

    # --- access ---------------------------------------------------------
    @property
    def layer_order(self):
        return [l.name for l in self.layers]

    def layer(self, name):
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"{self.name} has no layer '{name}' "
                       f"(layers: {self.layer_order})")

    def index(self, name):
        return self.layer_order.index(self.layer(name).name)

    def materials_used(self):
        return sorted({l.material for l in self.layers if l.material})

    # --- serialisation ----------------------------------------------------
    def to_json(self):
        return {
            "name": self.name,
            "parent": self.parent,
            "description": self.description,
            "ops": copy.deepcopy(self.ops),
            "layers": [l.to_json() for l in self.layers],
            "instrument": self.instrument.to_json(),
            **({"overrides": copy.deepcopy(self.overrides)} if self.overrides else {}),
            **({"seed": copy.deepcopy(self.seed)} if self.seed else {}),
            **({"links": copy.deepcopy(self.links)} if self.links else {}),
        }

    @classmethod
    def from_json(cls, d):
        return cls(name=d["name"], layers=d["layers"],
                   instrument=d.get("instrument", {}),
                   parent=d.get("parent"), ops=d.get("ops", []),
                   description=d.get("description", ""),
                   overrides=d.get("overrides", []), seed=d.get("seed"),
                   links=d.get("links"))

    def physics_hash(self):
        """Hash of what determines the objectives (layers + instrument)."""
        d = {"layers": [l.to_json() for l in self.layers],
             "instrument": self.instrument.to_json()}
        if self.overrides:          # absent → hash unchanged for older recipes
            d["overrides"] = self.overrides
        if self.seed:
            d["seed"] = {"model": self.seed["model"], "runs": self.seed.get("runs")}
            for k in ("layers", "instrument"):     # absent → older hashes unchanged
                if k in self.seed:
                    d["seed"][k] = self.seed[k]
        if self.links:
            d["links"] = self.links
        payload = json.dumps(d, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # --- bridge to batch.generate_batch_models ----------------------------
    def to_generate_kwargs(self, material_arrays):
        """
        material_arrays : {material_name: ndarray [E, Re, Im]}

        Returns kwargs for batch.generate_batch_models (minus data_dict,
        energy_list, sample_name).  SLD arrays are keyed by *layer* name so
        several layers can share one material.
        """
        sld_arrays, constants, lp, offsets = {}, {}, {}, {}
        for l in self.layers:
            if l.is_constant:
                constants[l.name] = dict(l.constant_sld)
            else:
                if l.material not in material_arrays:
                    raise KeyError(f"Layer '{l.name}' uses material "
                                   f"'{l.material}', which is not registered.")
                sld_arrays[l.name] = material_arrays[l.material][:, :3]
            lp[l.name] = l.layer_params()
            if l.sld_offset:
                offsets[l.name] = dict(l.sld_offset)
        kw = dict(material_sld_arrays=sld_arrays, constant_materials=constants,
                  base_layer_params=lp, layer_order=self.layer_order,
                  sld_offset_bounds=offsets)
        kw.update(self.instrument.generate_kwargs())
        return kw


# ---------------------------------------------------------------------------
# Derivation ops  (each returns a NEW recipe; the input is never mutated)
# ---------------------------------------------------------------------------

_LAYER_FIELDS = {f.name for f in fields(LayerSpec)} - {"name"}
_INSTR_FIELDS = {f.name for f in fields(InstrumentSpec)}


def _check_layer_fields(kw):
    bad = set(kw) - _LAYER_FIELDS
    if bad:
        raise ValueError(f"Unknown layer field(s) {sorted(bad)}; "
                         f"allowed: {sorted(_LAYER_FIELDS)}")


def add_layer(r, name, above=None, below=None, like=None, **kw):
    """
    Insert a new layer directly above or below an existing one.
    like=<layer> copies that layer's settings first; kw then override.
    material defaults to like's material, else to `name`.
    """
    if (above is None) == (below is None):
        raise ValueError("add_layer needs exactly one of above= / below=")
    if name in r.layer_order:
        raise ValueError(f"Layer '{name}' already exists in {r.name}")
    _check_layer_fields(kw)
    base = r.layer(like).to_json() if like else {}
    base.pop("name", None)
    base.update(kw)
    base.setdefault("material", name)
    new = LayerSpec(name=name, **base)
    out = copy.deepcopy(r)
    i = out.index(above) if above else out.index(below) + 1
    out.layers.insert(i, new)
    return out


def remove_layer(r, name):
    out = copy.deepcopy(r)
    out.layers.pop(out.index(name))
    out.overrides = [o for o in out.overrides if o["layer"] != name]
    if out.links:
        out.links["params"].pop(name, None)
        out.links = _norm_links(out.links)
    return out


def move_layer(r, name, above=None, below=None):
    if (above is None) == (below is None):
        raise ValueError("move_layer needs exactly one of above= / below=")
    out = copy.deepcopy(r)
    lay = out.layers.pop(out.index(name))
    i = out.index(above) if above else out.index(below) + 1
    out.layers.insert(i, lay)
    return out


def set_layer(r, layer, **kw):
    """Change values / bounds / offsets of one layer (None clears a bound)."""
    _check_layer_fields(kw)
    out = copy.deepcopy(r)
    l = out.layer(layer)
    d = l.to_json()
    d.update(kw)
    out.layers[out.index(layer)] = LayerSpec.from_json(
        {k: v for k, v in d.items() if v is not None})
    return out


def set_material(r, layer, material):
    return set_layer(r, layer, material=material)


def rename_layer(r, old, new):
    if new in r.layer_order:
        raise ValueError(f"Layer '{new}' already exists")
    out = copy.deepcopy(r)
    out.layer(old).name = new
    for o in out.overrides:
        if o["layer"] == old:
            o["layer"] = new
    if out.links and old in out.links["params"]:
        out.links["params"][new] = out.links["params"].pop(old)
    return out


def set_instrument(r, **kw):
    bad = set(kw) - _INSTR_FIELDS
    if bad:
        raise ValueError(f"Unknown instrument field(s) {sorted(bad)}")
    out = copy.deepcopy(r)
    d = out.instrument.to_json()
    d.update(kw)
    out.instrument = InstrumentSpec.from_json(d)
    return out


# --- energy links (global fits) ---------------------------------------------

LINKABLE = ("thickness", "roughness")


def _norm_links(d):
    """{"params": {layer: [...]}, "energies": None | [E | {"erange": [lo, hi]}]}.
    An erange entry is resolved to data energies by the project on save."""
    if not d:
        return None
    params = {}
    for layer, ps in (d.get("params") or {}).items():
        ps = [ps] if isinstance(ps, str) else list(ps)
        bad = set(ps) - set(LINKABLE)
        if bad:
            raise ValueError(f"cannot link {sorted(bad)} of '{layer}'; "
                             f"linkable: {list(LINKABLE)}")
        ps = [p for p in LINKABLE if p in ps]
        if ps:
            params[str(layer)] = ps
    en = d.get("energies")
    if en is not None:
        fl = sorted({float(e) for e in en if not isinstance(e, dict)})
        rg = [{"erange": sorted(float(x) for x in e["erange"])}
              for e in en if isinstance(e, dict)]
        en = fl + rg
    if not params and en is None:
        return None
    return {"params": params, "energies": en}


def link(r, layer, params=LINKABLE):
    """
    Share `layer`'s thickness and/or roughness across all fitted energies
    (one parameter for the whole global fit).  SLD, scale and bkg stay per
    energy.  A recipe with links is fitted as one global objective over its
    link energy set (see link_energies).
    """
    r.layer(layer)
    out = copy.deepcopy(r)
    cur = copy.deepcopy(out.links) or {"params": {}, "energies": None}
    ps = [params] if isinstance(params, str) else list(params)
    cur["params"][layer] = sorted(set(cur["params"].get(layer, [])) | set(ps),
                                  key=LINKABLE.index)
    out.links = _norm_links(cur)
    return out


def unlink(r, layer=None, params=None):
    """Remove links (all, one layer's, or some of its params)."""
    out = copy.deepcopy(r)
    if not out.links:
        return out
    cur = out.links
    for l in ([layer] if layer else list(cur["params"])):
        if l not in cur["params"]:
            continue
        keep = [] if params is None else [p for p in cur["params"][l]
                                          if p not in ([params] if isinstance(params, str) else params)]
        if keep:
            cur["params"][l] = keep
        else:
            del cur["params"][l]
    out.links = _norm_links(cur)
    return out


def link_energies(r, energies=None, erange=None):
    """
    Fix the energy set of the global fit: energies=[...] and/or
    erange=(lo, hi) (matched to data energies by the project).  Both None →
    all data energies.  The set is part of the model so global χ² / log Z are
    comparable between runs of the same model.
    """
    out = copy.deepcopy(r)
    cur = copy.deepcopy(out.links) or {"params": {}, "energies": None}
    if energies is None and erange is None:
        cur["energies"] = None
    else:
        es = list(energies or [])
        if erange is not None:
            es.append({"erange": [float(x) for x in erange]})
        cur["energies"] = es
    out.links = _norm_links(cur)
    return out


# --- per-energy overrides ----------------------------------------------------

_OVR_FIELDS = ("sld_real", "sld_imag", "sld_offset", "thickness", "roughness")


def _norm_override(o):
    o = dict(o)
    bad = set(o) - set(_OVR_FIELDS) - {"energy", "layer"}
    if bad:
        raise ValueError(f"unknown override field(s) {sorted(bad)}; "
                         f"allowed: {list(_OVR_FIELDS)}")
    out = {"energy": float(o["energy"]), "layer": str(o["layer"])}
    for k in ("sld_real", "sld_imag", "thickness", "roughness"):
        if o.get(k) is not None:
            out[k] = float(o[k])
    if o.get("sld_offset") is not None:
        out["sld_offset"] = _offset(o["sld_offset"])
    if len(out) == 2:
        raise ValueError("an override needs at least one of " + ", ".join(_OVR_FIELDS))
    return out


def set_energy_override(r, layer, energies=None, energy=None, **kw):
    """
    At the given energy/energies only, start `layer` from other values.
    sld_real / sld_imag: new initial SLD; bounds are re-centred on it using
    sld_offset (default: the layer's own sld_offset).  sld_offset alone (no
    sld_real/sld_imag for that part) re-windows the bounds around the
    tabulated SLD and keeps the current start value.  thickness / roughness:
    new initial values within the layer's usual bounds.  Merges with an
    existing override for the same energy and layer.
    """
    r.layer(layer)                                        # must exist
    es = [energy] if energy is not None else list(energies or [])
    if not es:
        raise ValueError("set_energy_override needs energy= or energies=")
    out = copy.deepcopy(r)
    for e in es:
        new = _norm_override({"energy": e, "layer": layer, **kw})
        cur = [o for o in out.overrides
               if o["layer"] == layer and abs(o["energy"] - new["energy"]) < 1e-3]
        if cur:
            cur[0].update({k: v for k, v in new.items() if k not in ("energy", "layer")})
        else:
            out.overrides.append(new)
    out.overrides.sort(key=lambda o: (o["energy"], o["layer"]))
    return out


def clear_energy_override(r, layer=None, energies=None, energy=None):
    """Remove overrides (all, one layer's, and/or at given energies)."""
    es = [energy] if energy is not None else energies
    out = copy.deepcopy(r)
    out.overrides = [o for o in out.overrides
                     if not ((layer is None or o["layer"] == layer)
                             and (es is None or any(abs(o["energy"] - float(e)) < 1e-3
                                                    for e in es)))]
    return out


def seed_from_fit(r, model, criteria="best", layers=None, instrument=True):
    """
    Start this recipe's *varying* parameters from `model`'s fitted values
    (per energy, matched by exact parameter name).  The project resolves which
    run is used at each energy when the recipe is saved and records it, so the
    start point is reproducible.  Bounds are unchanged; out-of-bounds seeds
    are clipped.  model=None removes the seed.

    layers=[...] seeds only those layers' parameters ("<layer> - ..."); the
    instrument parameters (scale, bkg, ...) follow `instrument`.  Other layers
    start from the recipe values.
    """
    out = copy.deepcopy(r)
    if model is None:
        out.seed = None
        return out
    out.seed = {"model": str(model), "criteria": criteria}
    if layers is not None:
        for l in layers:
            r.layer(l)                                     # must exist
        out.seed["layers"] = [str(l) for l in layers]
        out.seed["instrument"] = bool(instrument)
    elif not instrument:
        out.seed["instrument"] = False
    return out


def seed_selects(seed, pname):
    """True if a seed spec covers parameter `pname`."""
    if " - " not in pname:                                # scale, bkg, dq ...
        return seed.get("instrument", True)
    layers = seed.get("layers")
    return layers is None or pname.split(" - ", 1)[0] in layers


def apply_seed(values, objective, seed=None):
    """
    values: {param_name: fitted value} for one energy.  Sets varying
    parameters of `objective` (clipped into bounds) → (n_seeded, clipped list).
    seed (the recipe's seed spec) restricts which parameters are set.
    """
    n, clipped = 0, []
    for p in objective.parameters.flattened():
        if not p.vary or p.name not in values:
            continue
        if seed is not None and not seed_selects(seed, p.name):
            continue
        v = float(values[p.name])
        lo, hi = getattr(p.bounds, "lb", -np.inf), getattr(p.bounds, "ub", np.inf)
        if v < lo or v > hi:
            clipped.append((p.name, v, float(lo), float(hi)))
            v = float(min(max(v, lo), hi))
        p.value = v
        n += 1
    return n, clipped


def apply_overrides(r, energy, objective, rebound=True):
    """
    Apply r's overrides at `energy` to a built objective (in place).
    rebound=False skips bounds-only SLD re-windowing, which must run exactly
    once (it reads the tabulated centre back from the current bounds).
    """
    params = {p.name: p for p in objective.parameters.flattened()}
    done = []
    for o in r.overrides_at(energy):
        lay = r.layer(o["layer"])
        off = o.get("sld_offset") or lay.sld_offset or {}
        for part, key, pname in (("real", "sld_real", f"{lay.name} - sld"),
                                 ("imag", "sld_imag", f"{lay.name} - isld")):
            if key not in o:
                if (rebound and part in (o.get("sld_offset") or {})
                        and part in (lay.sld_offset or {})):
                    # bounds-only: re-window around the tabulated SLD, keep the
                    # current (possibly seeded) start value
                    p = params[pname]
                    lo, hi, vary = o["sld_offset"][part]
                    centre = float(p.bounds.ub) - lay.sld_offset[part][1]
                    lo, hi = centre + lo, centre + hi
                    if part == "imag":
                        lo = max(0.0, lo)
                    v = float(min(max(p.value, lo), hi))
                    p.setp(value=v, bounds=(lo, hi), vary=vary)
                    done.append(f"{pname} bounds=[{lo:g}, {hi:g}]")
                continue
            p = params[pname]
            v = o[key]
            if part in off:
                lo, hi, vary = off[part]
                lo, hi = v + lo, v + hi
                if part == "imag":
                    lo = max(0.0, lo)
                    v = max(0.0, v)
                p.setp(value=v, bounds=(lo, hi), vary=vary)
            elif getattr(lay, f"sld_{part}_bounds") is not None:   # absolute bounds
                p.value = v
            else:                                           # fixed SLD
                p.setp(value=v, vary=False)
            done.append(f"{pname}={v:g}")
        for key, pname in (("thickness", f"{lay.name} - thick"),
                           ("roughness", f"{lay.name} - rough")):
            if key in o:
                params[pname].value = o[key]
                done.append(f"{pname}={o[key]:g}")
    return done


OPS = {
    "add_layer": add_layer,
    "remove_layer": remove_layer,
    "move_layer": move_layer,
    "set_layer": set_layer,
    "set_bounds": set_layer,       # alias
    "set_material": set_material,
    "rename_layer": rename_layer,
    "set_instrument": set_instrument,
    "set_energy_override": set_energy_override,
    "clear_energy_override": clear_energy_override,
    "seed_from_fit": seed_from_fit,
    "link": link,
    "unlink": unlink,
    "link_energies": link_energies,
}


def apply_op(r, op):
    op = dict(op)
    kind = op.pop("op")
    if kind not in OPS:
        raise ValueError(f"Unknown op '{kind}'; known: {sorted(OPS)}")
    return OPS[kind](r, **op)


def derive(parent, new_name, ops, description=""):
    """Apply ops (list of {'op': ..., ...}) to parent → new recipe with lineage."""
    out = copy.deepcopy(parent)
    for op in ops:
        out = apply_op(out, op)
    out.name = new_name
    out.parent = parent.name
    out.ops = copy.deepcopy(list(ops))
    out.description = description
    return out


# ---------------------------------------------------------------------------
# Validation, diff, display
# ---------------------------------------------------------------------------

def validate(r, known_materials=None):
    """Return (errors, warnings) lists of strings."""
    errors, warnings = [], []
    names = r.layer_order
    if len(set(names)) != len(names):
        errors.append(f"duplicate layer names: {names}")
    if not r.layers:
        errors.append("recipe has no layers")
        return errors, warnings
    if not r.layers[0].is_constant:
        warnings.append(f"top layer '{names[0]}' is not a constant (air) layer")
    if known_materials is not None:
        for l in r.layers:
            if l.material and l.material not in known_materials:
                errors.append(f"layer '{l.name}': material '{l.material}' "
                              f"is not registered")
    for l in r.layers:
        for f, val in (("thickness_bounds", l.thickness),
                       ("roughness_bounds", l.roughness)):
            b = getattr(l, f)
            if b is None:
                continue
            lo, hi, vary = b
            if vary and not hi > lo:
                errors.append(f"layer '{l.name}' {f}={b}: vary=True needs hi > lo")
            elif vary and not lo <= val <= hi:
                errors.append(f"layer '{l.name}': initial "
                              f"{f.split('_')[0]}={val} outside {f}={b}")
        for part, b in (l.sld_offset or {}).items():
            lo, hi, vary = b
            if vary and not hi > lo:
                errors.append(f"layer '{l.name}' sld_offset[{part}]={b}: "
                              f"vary=True needs hi > lo")
            elif vary and not lo <= 0 <= hi:
                warnings.append(f"layer '{l.name}' sld_offset[{part}]={b} "
                                f"excludes the tabulated SLD")
        if (not l.is_constant and l.sld_offset is None
                and l.sld_real_bounds is None and l.sld_imag_bounds is None):
            pass
    for o in r.overrides:
        if o["layer"] not in names:
            errors.append(f"override at {o['energy']:g} eV names missing layer "
                          f"'{o['layer']}'")
            continue
        l = r.layer(o["layer"])
        for key, b in (("thickness", l.thickness_bounds), ("roughness", l.roughness_bounds)):
            if key in o and b is not None and b[2] and not b[0] <= o[key] <= b[1]:
                errors.append(f"override {o['layer']} {key}={o[key]:g} at "
                              f"{o['energy']:g} eV outside {b[:2]}")
        if l.is_constant and ("sld_real" in o or "sld_imag" in o):
            errors.append(f"override changes the SLD of constant layer '{l.name}'")
        bounds_only = [k for k in (o.get("sld_offset") or {})
                       if f"sld_{k}" not in o and k not in (l.sld_offset or {})]
        if bounds_only:
            errors.append(f"override at {o['energy']:g} eV re-bounds {o['layer']} "
                          f"{bounds_only} but the layer has no sld_offset for it")
    if r.links:
        for layer, ps in r.links["params"].items():
            if layer not in names:
                errors.append(f"link names missing layer '{layer}'")
                continue
            l = r.layer(layer)
            for pn in ps:
                b = getattr(l, f"{pn}_bounds")
                if l.is_constant or b is None or not b[2]:
                    warnings.append(f"link {layer} {pn}: parameter is fixed, "
                                    f"linking has no effect")
                if any(o["layer"] == layer and pn in o for o in r.overrides):
                    errors.append(f"link {layer} {pn}: a per-energy override sets "
                                  f"its start value; clear it or unlink")
        if r.links["params"] and r.links.get("energies") is not None \
                and len(r.links["energies"]) < 2:
            errors.append("a global fit needs at least 2 link energies")
    return errors, warnings


def _fmt_b(b):
    if b is None:
        return "fixed"
    lo, hi, vary = b
    return f"[{lo:g}, {hi:g}]" + ("" if vary else " (fixed)")


def _fmt_off(o):
    if not o:
        return "fixed"
    parts = []
    for k in ("real", "imag"):
        if k in o:
            lo, hi, vary = o[k]
            s = f"±{hi:g}" if lo == -hi else f"[{lo:+g}, {hi:+g}]"
            parts.append(f"{k[:2]} {s}" + ("" if vary else " (fixed)"))
    return ", ".join(parts)


def layer_table(r):
    """Plain-text table of a recipe's layers (top → bottom)."""
    rows = [("layer", "material", "thick", "thick bounds", "rough",
             "rough bounds", "SLD")]
    for l in r.layers:
        if l.is_constant:
            sld = f"const {l.constant_sld['real']:g}+{l.constant_sld['imag']:g}j"
        elif l.sld_real_bounds or l.sld_imag_bounds:
            sld = f"abs re {_fmt_b(l.sld_real_bounds)}, im {_fmt_b(l.sld_imag_bounds)}"
        else:
            sld = _fmt_off(l.sld_offset)
        rows.append((l.name, l.material or "-", f"{l.thickness:g}",
                     _fmt_b(l.thickness_bounds), f"{l.roughness:g}",
                     _fmt_b(l.roughness_bounds), sld))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * x for x in w))
    ins = r.instrument
    lines.append(
        f"instrument: scale={ins.scale:g} x{list(ins.scale_bounds)} "
        f"{'vary' if ins.vary_scale else 'fixed'} | bkg="
        f"{'auto' if ins.bkg is None else f'{ins.bkg:g}'} x{list(ins.bkg_bounds)} "
        f"{'vary' if ins.vary_bkg else 'fixed'} | dq={ins.dq:g} "
        f"{list(ins.dq_bounds) if ins.vary_dq else '(fixed)'}")
    for o in r.overrides:
        lines.append("override " + _fmt_override(o, r))
    if r.seed:
        lines.append("seed " + _fmt_seed(r.seed))
    if r.links:
        lines.append("links " + _fmt_links(r.links))
    return "\n".join(lines)


def _fmt_links(lk):
    ps = "; ".join(f"{l} {'+'.join(v)}" for l, v in lk["params"].items()) or "(none)"
    en = lk.get("energies")
    en = ("all data energies" if en is None else
          f"{len(en)} energies " + ", ".join(f"{e:g}" if not isinstance(e, dict)
                                             else f"{e['erange'][0]:g}–{e['erange'][1]:g}"
                                             for e in en))
    return f"shared across energies: {ps}  [global fit over {en}]"


def _fmt_seed(sd):
    runs = sd.get("runs") or {}
    what = "fitted values"
    if sd.get("layers") is not None:
        what = ("fitted values of " + ", ".join(sd["layers"])
                + (" + instrument" if sd.get("instrument", True) else ""))
    elif not sd.get("instrument", True):
        what = "fitted layer values (not instrument)"
    return (f"start from {sd['model']} {what} ({sd.get('criteria', 'best')} run; "
            f"{len(runs)} energies" + ("" if runs else ", resolved on save") + ")")


def _fmt_override(o, r=None):
    parts = []
    if "sld_real" in o or "sld_imag" in o:
        off = o.get("sld_offset") or (r.layer(o["layer"]).sld_offset
                                      if r is not None and o["layer"] in r.layer_order
                                      else None)
        sld = ", ".join(f"{k[4:]}={o[k]:g}" for k in ("sld_real", "sld_imag") if k in o)
        parts.append(f"SLD start {sld} ({_fmt_off(off)})")
    elif o.get("sld_offset"):
        parts.append(f"SLD bounds tab {_fmt_off(o['sld_offset'])}")
    for k in ("thickness", "roughness"):
        if k in o:
            parts.append(f"{k}={o[k]:g}")
    return f"@ {o['energy']:g} eV  {o['layer']}: " + "; ".join(parts)


def diff(a, b):
    """Human-readable list of differences a → b."""
    out = []
    if a.layer_order != b.layer_order:
        out.append(f"layer order: {' / '.join(a.layer_order)}  →  "
                   f"{' / '.join(b.layer_order)}")
    an, bn = set(a.layer_order), set(b.layer_order)
    for n in b.layer_order:
        if n not in an:
            i = b.index(n)
            where = (f"above {b.layers[i + 1].name}" if i + 1 < len(b.layers)
                     else f"below {b.layers[i - 1].name}")
            out.append(f"+ {n} ({where}): {_layer_summary(b.layer(n))}")
    for n in a.layer_order:
        if n not in bn:
            out.append(f"- {n}")
    for n in b.layer_order:
        if n in an:
            la, lb = a.layer(n).to_json(), b.layer(n).to_json()
            for k in sorted(set(la) | set(lb)):
                if la.get(k) != lb.get(k):
                    out.append(f"~ {n}.{k}: {la.get(k)} → {lb.get(k)}")
    ia, ib = a.instrument.to_json(), b.instrument.to_json()
    for k in ia:
        if ia[k] != ib[k]:
            out.append(f"~ instrument.{k}: {ia[k]} → {ib[k]}")
    ka = {(o["energy"], o["layer"]): o for o in a.overrides}
    kb = {(o["energy"], o["layer"]): o for o in b.overrides}
    for k in sorted(set(ka) | set(kb)):
        if k not in ka:
            out.append("+ override " + _fmt_override(kb[k], b))
        elif k not in kb:
            out.append("- override " + _fmt_override(ka[k], a))
        elif ka[k] != kb[k]:
            out.append("~ override " + _fmt_override(kb[k], b))
    if a.links != b.links:
        out.append(("~ links " + _fmt_links(b.links)) if b.links else "- links")
    _sk = lambda sd: {k: v for k, v in (sd or {}).items() if k != "runs"}
    if _sk(a.seed) != _sk(b.seed):
        out.append(("~ seed " + _fmt_seed(b.seed)) if b.seed else "- seed")
    return out or ["(no differences)"]


def _layer_summary(l):
    return (f"material={l.material}, thick={l.thickness:g} "
            f"{_fmt_b(l.thickness_bounds)}, rough={l.roughness:g} "
            f"{_fmt_b(l.roughness_bounds)}, SLD {_fmt_off(l.sld_offset)}")
