"""
project.py
----------
FitProject: a directory that is the source of truth for one sample's fitting
campaign.

    <project>/
        project.json        data source, materials, model recipes (+ lineage)
        fits.h5             h5io layout /{sample}/{E}/{model}/run_N/
        figures/  logs/  jobs/  uncertainty/
"""

import contextlib
import io
import json
import os
import tempfile
from datetime import datetime

import numpy as np

from . import recipe as R
from .materials import MaterialSource, resolve

SCHEMA_VERSION = 1

DEFAULT_SETTINGS = {
    "fit":   {"popsize": 500, "n_generations": 5000, "tol": 1e-4,
              "patience": 5, "n_restarts": 1, "normalize": False, "seed": 0},
    "nuts":  {"n_chains": 4, "n_warmup": 500, "samples_per_chunk": 200,
              "min_samples": 500, "max_samples": 5000, "rhat_threshold": 1.05,
              "ess_min": 200, "normalize": True},
    "jaxns": {"max_samples": 100_000, "num_live_points": 500, "seed": 0,
              "n_posterior_samples": 2000, "normalize": True},
}


class ProjectError(RuntimeError):
    pass


@contextlib.contextmanager
def _quiet(verbose):
    if verbose:
        yield None
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class FitProject:
    # ------------------------------------------------------------------
    # create / open / save
    # ------------------------------------------------------------------
    def __init__(self, path, cfg):
        self.path = os.path.abspath(path)
        self.cfg = cfg
        self._data = None
        self._arrays = {}

    @classmethod
    def create(cls, path, sample_name, data_folder, file_type="smoothed",
               q_max=None, h5_name="fits.h5", description=""):
        path = os.path.abspath(path)
        if os.path.exists(os.path.join(path, "project.json")):
            raise ProjectError(f"{path} already contains a project.json")
        os.makedirs(path, exist_ok=True)
        cfg = {
            "schema_version": SCHEMA_VERSION,
            "sample_name": sample_name,
            "description": description,
            "created": datetime.now().isoformat(timespec="seconds"),
            "data": {"folder": os.path.abspath(data_folder),
                     "file_type": file_type, "q_max": q_max, "energies": []},
            "h5": h5_name,
            "settings": json.loads(json.dumps(DEFAULT_SETTINGS)),
            "materials": {},
            "models": {},
        }
        proj = cls(path, cfg)
        _, energies = proj.load_data(verbose=False)
        if not energies:
            raise ProjectError(f"no '{file_type}' reflectivity files found in "
                               f"{data_folder}")
        cfg["data"]["energies"] = [float(e) for e in energies]
        proj._ensure_dirs()
        proj.save()
        return proj

    @classmethod
    def open(cls, path):
        path = os.path.abspath(path)
        fp = os.path.join(path, "project.json")
        if not os.path.exists(fp):
            raise ProjectError(f"no project.json in {path}")
        with open(fp) as fh:
            cfg = json.load(fh)
        proj = cls(path, cfg)
        proj._ensure_dirs()
        return proj

    def save(self):
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".project.", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.cfg, fh, indent=2)
        os.replace(tmp, os.path.join(self.path, "project.json"))

    def _ensure_dirs(self):
        for d in ("figures", "logs", "jobs", "uncertainty"):
            os.makedirs(os.path.join(self.path, d), exist_ok=True)

    # ------------------------------------------------------------------
    # paths / basic properties
    # ------------------------------------------------------------------
    @property
    def sample_name(self):
        return self.cfg["sample_name"]

    @property
    def h5_path(self):
        return os.path.join(self.path, self.cfg["h5"])

    @property
    def energies(self):
        return list(self.cfg["data"]["energies"])

    def dir(self, name):
        return os.path.join(self.path, name)

    def settings(self, kind):
        return dict(self.cfg.get("settings", {}).get(kind, DEFAULT_SETTINGS[kind]))

    def select_energies(self, energies=None, erange=None):
        """Match requested energies (or an inclusive range) to data energies."""
        avail = np.array(self.energies)
        if energies is None and erange is None:
            return list(avail)
        out = []
        if energies is not None:
            for e in energies:
                i = int(np.argmin(np.abs(avail - float(e))))
                if abs(avail[i] - float(e)) > 1e-3:
                    raise ProjectError(f"no data at {e} eV; available: "
                                       f"{', '.join(f'{x:g}' for x in avail)}")
                out.append(float(avail[i]))
        if erange is not None:
            lo, hi = sorted(map(float, erange))
            out += [float(e) for e in avail if lo - 1e-6 <= e <= hi + 1e-6]
        out = sorted(set(out))
        if not out:
            raise ProjectError(f"no data energies selected by {energies} / {erange}")
        return out

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def load_data(self, energies=None, verbose=False):
        """→ (data_dict {E: ReflectDataset}, energy_list), cached per process."""
        if self._data is None:
            from batch import import_batch_reflectivity
            d = self.cfg["data"]
            with _quiet(verbose):
                self._data = import_batch_reflectivity(
                    d["folder"], file_type=d["file_type"], q_max=d["q_max"])
        data_dict, elist = self._data
        if energies is None:
            return data_dict, list(elist)
        sel = self.select_energies(energies)
        return {e: data_dict[e] for e in sel}, sel

    # ------------------------------------------------------------------
    # materials
    # ------------------------------------------------------------------
    def add_material(self, name, source=None, overwrite=False, **kw):
        """add_material('SOG', kind='file', path=...) or with a MaterialSource."""
        if source is None:
            source = MaterialSource(**kw)
        elif isinstance(source, dict):
            source = MaterialSource.from_json(source)
        if name in self.cfg["materials"] and not overwrite:
            if self.cfg["materials"][name] == source.to_json():
                return source
            raise ProjectError(f"material '{name}' already registered "
                               f"(pass overwrite=True to replace)")
        arr = resolve(source, self.energies, self.path)       # fail early
        emin, emax = arr[:, 0].min(), arr[:, 0].max()
        out = [e for e in self.energies if not emin <= e <= emax]
        if out:
            print(f"warning: material '{name}' covers {emin:g}–{emax:g} eV; "
                  f"{len(out)} data energies will be extrapolated")
        self.cfg["materials"][name] = source.to_json()
        self._arrays.pop(name, None)
        self.save()
        return source

    def material(self, name):
        try:
            return MaterialSource.from_json(self.cfg["materials"][name])
        except KeyError:
            raise ProjectError(f"material '{name}' not registered; known: "
                               f"{sorted(self.cfg['materials'])}") from None

    def material_array(self, name):
        if name not in self._arrays:
            self._arrays[name] = resolve(self.material(name), self.energies, self.path)
        return self._arrays[name]

    def material_arrays(self, names=None):
        names = sorted(self.cfg["materials"]) if names is None else names
        return {n: self.material_array(n) for n in names}

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    @property
    def model_names(self):
        return list(self.cfg["models"])

    def get_model(self, name):
        try:
            return R.ModelRecipe.from_json(self.cfg["models"][name])
        except KeyError:
            raise ProjectError(f"no model '{name}'; known: {self.model_names}") from None

    def has_fits(self, name):
        """True if the h5 holds any run for this model."""
        if not os.path.exists(self.h5_path):
            return False
        import h5py
        with h5py.File(self.h5_path, "r") as f:
            if self.sample_name not in f:
                return False
            g = f[self.sample_name]
            return any(isinstance(g[k], h5py.Group) and name in g[k] for k in g)

    def add_model(self, recipe, overwrite=False):
        """Validate and store a recipe.  Returns the list of warnings."""
        if recipe.name in self.cfg["models"]:
            if not overwrite:
                raise ProjectError(f"model '{recipe.name}' exists "
                                   f"(overwrite=True to replace, or derive a new one)")
            if self.has_fits(recipe.name):
                raise ProjectError(
                    f"model '{recipe.name}' already has fits in the h5 and is "
                    f"frozen; derive a new model instead")
        if recipe.parent and recipe.parent not in self.cfg["models"]:
            raise ProjectError(f"parent '{recipe.parent}' is not in the project")
        if recipe.seed and not recipe.seed.get("runs"):
            recipe.seed["runs"] = self._resolve_seed(recipe.seed)
        if recipe.links and recipe.links.get("energies") is not None:
            recipe.links["energies"] = self._resolve_link_energies(recipe.links["energies"])
        errors, warnings = R.validate(recipe, known_materials=self.cfg["materials"])
        for o in recipe.overrides:
            if not any(abs(o["energy"] - e) < 1e-3 for e in self.energies):
                errors.append(f"override at {o['energy']:g} eV: no data at that energy")
        if errors:
            raise ProjectError(f"{recipe.name} is invalid:\n  " + "\n  ".join(errors))
        self.cfg["models"][recipe.name] = recipe.to_json()
        self.save()
        return warnings

    def new_model(self, name, layers, instrument=None, description="",
                  overwrite=False):
        r = R.ModelRecipe(name=name, layers=layers,
                          instrument=instrument or R.InstrumentSpec(),
                          description=description)
        self.add_model(r, overwrite=overwrite)
        return r

    def derive(self, parent, new_name, ops, description="", overwrite=False,
               seed_from=None, seed_criteria="best", seed_layers=None,
               seed_instrument=True):
        """
        Model new_name = parent + ops.  seed_from='ModelX' starts the new
        model's varying parameters from ModelX's fitted values (per energy);
        seed_layers=[...] limits that to those layers (+ instrument unless
        seed_instrument=False).  Without seed_from the parent's seed, if any,
        is dropped.  Returns the new recipe.
        """
        r = R.derive(self.get_model(parent), new_name, ops, description)
        if seed_from or not any(o.get("op") == "seed_from_fit" for o in ops):
            r = R.seed_from_fit(r, seed_from, seed_criteria, layers=seed_layers,
                                instrument=seed_instrument)
        self.add_model(r, overwrite=overwrite)
        return r

    def widen_stuck_sld(self, parent, new_name, layer, step=1.0, tol_pct=10.0,
                        parts=("real", "imag"), criteria="best", description="",
                        extra_ops=(), overwrite=False):
        """
        new_name = parent seeded from its own fit, with `layer`'s SLD window
        widened by `step` (1e-6 Å⁻²) on the side where the parent's fitted SLD
        ended near a bound, per energy.  "Near" = within tol_pct % of the window
        width (the red "near bound" dots in plot sld-energy).  Energies/parts
        not near a bound keep the parent's window.  Returns (recipe, report)
        with report = [(energy, part, value, lo, hi, side, new_lo, new_hi)].
        """
        from .analysis import param_table
        pr = self.get_model(parent)
        lay = pr.layer(layer)
        if not lay.sld_offset:
            raise ProjectError(f"{parent}: layer '{layer}' has no sld_offset to widen")
        if not self.has_fits(parent):
            raise ProjectError(f"'{parent}' has no fits to seed from")
        pt = param_table(self.h5_path, self.sample_name, [parent], criteria=criteria)
        names = {"real": f"{layer} - sld", "imag": f"{layer} - isld"}
        ops, report = list(extra_ops), []
        for e in sorted(pt.energy.unique()):
            ovr = next((o for o in pr.overrides_at(e) if o["layer"] == layer), {})
            base = ovr.get("sld_offset") or lay.sld_offset
            new_off, changed = {}, False
            for part in parts:
                if part not in lay.sld_offset:
                    continue
                row = pt[(pt.energy == e) & (pt.param == names[part])]
                if row.empty or not bool(row.vary.iloc[0]):
                    continue
                v, lb, ub = (float(row[c].iloc[0]) for c in ("value", "lb", "ub"))
                lo, hi, vary = base[part]
                span, tol = ub - lb, tol_pct / 100.0
                side = None
                if span > 0 and (ub - v) / span < tol:
                    side, hi = "upper", hi + step
                elif span > 0 and (v - lb) / span < tol:
                    if part == "imag" and lb <= 0:
                        side = "lower (at 0 clamp, not widened)"
                    else:
                        side, lo = "lower", lo - step
                new_off[part] = (lo, hi, vary)
                if side:
                    report.append((e, part, v, lb, ub, side,
                                   lb - (step if side == "lower" else 0),
                                   ub + (step if side == "upper" else 0)))
                    changed = changed or side in ("upper", "lower")
            if changed:
                ops.append({"op": "set_energy_override", "layer": layer, "energy": e,
                            "sld_offset": {**base, **new_off}})
        r = self.derive(parent, new_name, ops, description=description,
                        overwrite=overwrite, seed_from=parent, seed_criteria=criteria)
        return r, report

    def edit_model(self, name, ops):
        """Apply ops to an existing model in place (only while it has no fits)."""
        if self.has_fits(name):
            raise ProjectError(f"'{name}' already has fits and is frozen; derive a "
                               f"new model instead")
        r = self.get_model(name)
        before = R.ModelRecipe.from_json(r.to_json())
        for op in ops:
            r = R.apply_op(r, op)
        r.ops = list(r.ops) + list(ops)
        self.add_model(r, overwrite=True)
        return r, R.diff(before, r)

    def _resolve_link_energies(self, entries):
        """Link energy entries (floats / {"erange": [lo, hi]}) → data energies."""
        out = set()
        for e in entries:
            if isinstance(e, dict):
                out |= set(self.select_energies(erange=e["erange"]))
            else:
                out |= set(self.select_energies([e]))
        return sorted(out)

    def link_energy_list(self, model):
        """Energies of a linked model's global fit (all data energies if unset)."""
        r = model if isinstance(model, R.ModelRecipe) else self.get_model(model)
        en = (r.links or {}).get("energies")
        return self.energies if en is None else self._resolve_link_energies(en)

    def build_global(self, model, verbose=False, check=True):
        """
        Linked recipe → one refnx GlobalObjective over its link energies.

        Per-energy objectives are built exactly as build_objectives does
        (overrides, seed), then every linked thickness/roughness is replaced
        by ONE shared Parameter (start = median of the per-energy starts,
        bounds = the layer's bounds).  SLD, other layers, scale, bkg stay per
        energy.  Returns the dict GlobalModelsBatchBuilder /
        run_global_nested_sampling expect, plus 'checks'.
        """
        from refnx.analysis import GlobalObjective, Parameter
        r = model if isinstance(model, R.ModelRecipe) else self.get_model(model)
        if not r.is_linked:
            raise ProjectError(f"{r.name} has no links; use build_objectives")
        elist = self.link_energy_list(r)
        built = self.build_objectives(r, elist, verbose=verbose)
        objs = [built["objectives"][e] for e in elist]
        n_free_indep = sum(len(o.varying_parameters()) for o in objs)
        shared = {}
        attr = {"thickness": "thick", "roughness": "rough"}
        for layer, ps in r.links["params"].items():
            for pn in ps:
                slabs = []
                for e, o in zip(elist, objs):
                    hit = [c for c in o.model.structure if c.name == layer]
                    if len(hit) != 1:
                        raise ProjectError(f"{r.name} {e:g} eV: layer '{layer}' "
                                           f"found {len(hit)}× in the structure")
                    slabs.append(hit[0])
                old = [getattr(sl, attr[pn]) for sl in slabs]
                if not old[0].vary:
                    continue                                   # fixed: nothing to share
                lo, hi = float(old[0].bounds.lb), float(old[0].bounds.ub)
                v0 = float(np.clip(np.median([p.value for p in old]), lo, hi))
                sp = Parameter(v0, name=old[0].name, vary=True, bounds=(lo, hi))
                for sl in slabs:
                    setattr(sl, attr[pn], sp)
                shared[(layer, pn)] = sp
        gobj = GlobalObjective(objs)
        out = {"global_objective": gobj, "objectives": objs,
               "structures": [o.model.structure for o in objs],
               "models": [o.model for o in objs],
               "sample_names": [str(e) for e in elist], "energy_list": elist,
               "shared_params": shared, "linked": True,
               "seed_report": built["seed_report"]}
        if check:
            out["checks"] = self._check_global(out, n_free_indep)
        return out

    @staticmethod
    def _check_global(g, n_free_indep):
        """Parameter count, shared-object identity and χ² additivity."""
        N = len(g["energy_list"])
        n_expected = n_free_indep - (N - 1) * len(g["shared_params"])
        n_actual = len(g["global_objective"].varying_parameters())
        if n_actual != n_expected:
            raise ProjectError(f"global parameter count {n_actual} != expected "
                               f"{n_expected} (linking failed)")
        attr = {"thickness": "thick", "roughness": "rough"}
        for (layer, pn), sp in g["shared_params"].items():
            for st in g["structures"]:
                c = [x for x in st if x.name == layer][0]
                if getattr(c, attr[pn]) is not sp:
                    raise ProjectError(f"{layer} {pn} is not the shared parameter")
        chi_g = float(g["global_objective"].chisqr())
        chi_s = sum(float(o.chisqr()) for o in g["objectives"])
        if not np.isclose(chi_g, chi_s, rtol=1e-9):
            raise ProjectError(f"global χ² {chi_g} != Σ per-energy {chi_s}")
        return {"n_global": n_actual, "n_independent": n_free_indep,
                "n_shared": len(g["shared_params"]), "n_energies": N,
                "chi2_total": chi_g}

    def _resolve_seed(self, seed):
        """Pin the parent's run per energy (best/last/int) at save time."""
        import h5py
        from .analysis import _pick_run
        src = seed["model"]
        if not self.has_fits(src):
            raise ProjectError(f"cannot seed from '{src}': it has no fits in {self.h5_path}")
        runs = {}
        with h5py.File(self.h5_path, "r") as f:
            g = f[self.sample_name]
            for ek in g:
                try:
                    e = float(ek)
                except ValueError:
                    continue
                if src in g[ek]:
                    r = _pick_run(g[ek][src], seed.get("criteria", "best"))
                    if r is not None:
                        runs[str(e)] = r
        return dict(sorted(runs.items(), key=lambda kv: float(kv[0])))

    def _seed_values(self, seed, energies):
        """{E: {param: value}} from the pinned runs."""
        import h5py
        out = {}
        with h5py.File(self.h5_path, "r") as f:
            for ek, run in seed["runs"].items():
                e = float(ek)
                if not any(abs(e - x) < 1e-6 for x in energies):
                    continue
                pg = f[f"{self.sample_name}/{e}/{seed['model']}/{run}/parameters"]
                names = [n.decode() if isinstance(n, bytes) else n for n in pg["names"][()]]
                out[e] = dict(zip(names, pg["final_values"][()].tolist()))
        return out

    def remove_model(self, name):
        if self.has_fits(name):
            raise ProjectError(f"'{name}' has fits in the h5; refusing to remove")
        children = [m for m, d in self.cfg["models"].items() if d.get("parent") == name]
        if children:
            raise ProjectError(f"'{name}' is the parent of {children}")
        self.cfg["models"].pop(name)
        self.save()

    def lineage(self, name):
        chain = [name]
        while True:
            p = self.cfg["models"][chain[-1]].get("parent")
            if not p or p in chain:
                return chain[::-1]
            chain.append(p)

    def models_table(self):
        rows = [("model", "parent", "layers", "fits", "description")]
        for n in self.model_names:
            r = self.get_model(n)
            rows.append((n, r.parent or "-", "/".join(r.layer_order),
                         "yes" if self.has_fits(n) else "no", r.description))
        w = [max(len(r[i]) for r in rows) for i in range(4)]
        return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r[:4]))
                         + "  " + r[4] for r in rows)

    def show_model(self, name, diff_from="parent"):
        r = self.get_model(name)
        out = [f"{name}" + (f"  (from {r.parent})" if r.parent else "")
               + (f": {r.description}" if r.description else ""),
               R.layer_table(r)]
        base = r.parent if diff_from == "parent" else diff_from
        if base:
            out.append(f"changes vs {base}:")
            out += ["  " + s for s in R.diff(self.get_model(base), r)]
        return "\n".join(out)

    # ------------------------------------------------------------------
    # objectives
    # ------------------------------------------------------------------
    def build_objectives(self, model, energies=None, verbose=False):
        """
        Recipe → per-energy objectives via batch.generate_batch_models.

        Returns dict(objectives, structures, models, energies).  Raises if any
        requested energy failed to build (generate_batch_models only prints).
        """
        from batch import generate_batch_models
        r = model if isinstance(model, R.ModelRecipe) else self.get_model(model)
        data_dict, elist = self.load_data(energies)
        kw = r.to_generate_kwargs(self.material_arrays(r.materials_used()))
        with _quiet(verbose) as buf:
            models, structures, objectives = generate_batch_models(
                data_dict=data_dict, energy_list=elist,
                sample_name=self.sample_name, verbose=True, **kw)
        missing = [e for e in elist if e not in objectives]
        seed_report = None
        if not missing:
            # overrides first so seeds are clipped to the overridden bounds;
            # re-applied after seeding so value overrides still win
            for e in elist:
                R.apply_overrides(r, e, objectives[e])
            if r.seed:
                vals = self._seed_values(r.seed, elist)
                seed_report = {"model": r.seed["model"], "seeded": {}, "clipped": {},
                               "unseeded_energies": [e for e in elist if e not in vals]}
                for e, v in vals.items():
                    n, clipped = R.apply_seed(v, objectives[e], r.seed)
                    seed_report["seeded"][e] = n
                    if clipped:
                        seed_report["clipped"][e] = clipped
            for e in elist:
                R.apply_overrides(r, e, objectives[e], rebound=False)
        if missing:
            log = buf.getvalue() if buf is not None else ""
            errs = [l for l in log.splitlines() if "Error" in l]
            raise ProjectError(f"{r.name}: failed to build at {missing} eV\n"
                               + "\n".join(errs[:10]))
        return {"objectives": objectives, "structures": structures,
                "models": models, "energies": elist, "seed_report": seed_report}

    def load_best(self, model, energies=None, criteria="best", **kw):
        return load_best(self, model, energies, criteria, **kw)

    def unc_path(self, model, method, energy, sample=None):
        d = os.path.join(self.dir("uncertainty"), sample or self.sample_name, model)
        return os.path.join(d, f"{method}_{float(energy):.2f}eV.npz")


def load_best(project, model, energies=None, criteria="best", h5=None,
              sample=None, verbose=False):
    """
    Best-fit objectives for uncertainty analysis → dict(objectives, runs,
    source).

    Project model with fits in the project h5: the objectives are rebuilt from
    the recipe (so parameter identity / bounds are the recipe's) and the h5
    run's values are injected by exact parameter name.  The recomputed χ² must
    reproduce the stored chi_sq_final, otherwise ProjectError.

    h5=/sample= (e.g. a notebook h5) or a model with no recipe: falls back to
    h5io.load_h5_objectives (plain-SLD reconstruction).
    """
    import h5py
    from .analysis import _pick_run
    ext = h5 is not None
    h5 = h5 or project.h5_path
    sample = sample or project.sample_name
    if not os.path.exists(h5):
        raise ProjectError(f"no fits yet ({h5} missing)")

    with h5py.File(h5, "r") as f:
        g = f[sample]
        runs = {}
        for ek in g:
            try:
                e = float(ek)
            except ValueError:
                continue
            if model in g[ek]:
                r = _pick_run(g[ek][model], criteria)
                if r is not None:
                    runs[e] = r
    if not runs:
        raise ProjectError(f"no fits for {model} in {h5}")
    if energies is None:
        elist = sorted(runs)
    else:
        elist = []
        for e in energies:
            hit = [x for x in runs if abs(x - float(e)) < 1e-3]
            if not hit:
                raise ProjectError(f"{model} has no fit at {e} eV; fitted: "
                                   f"{', '.join(f'{x:g}' for x in sorted(runs))}")
            elist.append(hit[0])

    if ext or model not in project.cfg["models"]:
        from h5io import load_h5_objectives
        with _quiet(verbose):
            objs, _ = load_h5_objectives(h5, sample, model, criteria=criteria,
                                         energy_list=elist)
        return {"objectives": objs, "runs": {e: runs[e] for e in elist},
                "source": "h5-reconstructed", "h5": h5, "sample": sample}

    from .analysis import param_table
    recipe = project.get_model(model)
    objs = project.build_objectives(recipe, elist)["objectives"]
    pt = param_table(h5, sample, [model], criteria, energies=elist)
    import h5py as _h
    with _h.File(h5, "r") as f:
        for e in elist:
            rg = f[f"{sample}/{float(e)}/{model}/{runs[e]}"]
            rh = rg.attrs.get("recipe_hash")
            if rh is not None and rh != recipe.physics_hash():
                raise ProjectError(f"{model} {e:g} eV {runs[e]} was fitted with a "
                                   f"different recipe (hash {rh})")
            stored = float(rg.attrs["chi_sq_final"])
            vals = pt[pt.energy == e].set_index("param")["value"]
            obj = objs[e]
            for p in obj.parameters.flattened():
                if p.name in vals.index:
                    p.value = float(vals[p.name])
            chi = float(obj.chisqr())
            if not np.isclose(chi, stored, rtol=1e-6):
                raise ProjectError(f"{model} {e:g} eV: rebuilt χ²={chi:.6g} does not "
                                   f"reproduce stored {stored:.6g} ({runs[e]})")
    return {"objectives": objs, "runs": {e: runs[e] for e in elist},
            "source": "recipe", "h5": h5, "sample": sample}


def free_parameter_table(objective):
    """[(name, value, lo, hi)] for the varying parameters of an objective."""
    rows = []
    for p in objective.varying_parameters():
        b = p.bounds
        lo, hi = (getattr(b, "lb", np.nan), getattr(b, "ub", np.nan))
        rows.append((p.name, float(p.value), float(lo), float(hi)))
    return rows
