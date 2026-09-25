---
name: rsoxr-fit
description: Fit soft X-ray reflectivity (RSoXR) data with Refltools from plain-text requests — load a data folder, register optical constants, build and derive layer models ("add a SOG2 layer above SOG with ±0.5"), batch CMA-ES fits on GPU, χ²/SLD/reflectivity comparison plots, NUTS/JAXNS uncertainty and evidence. Use whenever the user talks about reflectivity fitting, RSoXR, layer models, SLD / optical constants, Brewer/LAM samples, fitting models, χ² comparisons, NUTS, JAXNS, BlackJAX, posterior or evidence for reflectivity.
---

# RSoXR fitting harness (Refltools/rsoxr_harness)

You translate the user's plain-text requests into calls on `rsoxr_harness`, a
project-directory front end to the Refltools pipeline
(`batch.generate_batch_models` → `batch_fit_selected_models_cmaes` → `h5io` →
`gpu_mcmc` / `gpu_nested_sampler`). **Never write ad-hoc notebook-style fitting
code when a harness call exists.** Package docs: module docstrings in
`/homes/dfs1/Refltools/rsoxr_harness/*.py`.

## Environment

```bash
PY=/homes/dfs1/mambaforge/envs/refnxlocal/bin/python
cd /homes/dfs1/Refltools            # package is importable from here
$PY -m rsoxr_harness <command> -p <project> ...
```
Python API (for model building): run `$PY - <<'EOF' ... EOF` from
`/homes/dfs1/Refltools` with `import sys; sys.path.insert(0, '.')` then
`from rsoxr_harness import FitProject, LayerSpec, InstrumentSpec, pm_offset, free_parameter_table`.
For quick non-GPU work prefix `JAX_PLATFORMS=cpu`. Pipe noisy output through
`grep -v "^Loaded\|^Interp"`.

## The project

One project directory per sample campaign, source of truth for everything:
`project.json` (data folder, materials, models + lineage), `fits.h5` (h5io
layout), `figures/`, `logs/`, `jobs/`, `uncertainty/`, `sld_exports/`.
Default location: next to the user's notebooks, e.g.
`/homes/dfs1/Reflectivity/Brewer/Nov2025/Brewer10_Fit_NB/Harness_Brewer10`.
Establish the active project once per session (ask if ambiguous) and reuse it.
Any older notebook h5 can be read by plot/fits/nuts/jaxns commands via
`--h5 <file> --sample <name>`.

## Intent → action

| User says | Do |
|---|---|
| "load the data in <folder>" | `ls` the folder first (file naming decides `--file-type`: `*smoothed.dat` → smoothed, `*raw.dat` → raw, otherwise `all`); sample name from the path; `init -p <proj> --sample S --data DIR --file-type T`; report #energies, q-range and point counts per energy, anything odd (short q ranges, R not normalised). |
| "use X for SOC" / "SOG from file F" / "SiO2 from formula at 2.2 g/cc" | `p.add_material(name, kind="file"\|"fit_csv", path=...)` or `kind="formula", formula=, density=`. Standard substrate files: `/homes/dfs1/Refltools/OC/Si_SLD.txt`, `/homes/dfs1/Refltools/OC/SiO2_Dean_SLD.txt`. If unspecified, look at the sample's CMAES notebook (cell with `material_sld_arrays`) and propose those, stated as assumptions. |
| "build Model1: air / SOG 335 ±5 / SOC / SiO2 / Si, SOG SLD ±3" | `p.new_model("Model1", [LayerSpec(...), ...], InstrumentSpec(...), description=...)` — show `p.show_model` before saving (see protocol). |
| "Model2 = Model1 + SOG2 layer above SOG, 15 Å (10–20)" | `p.derive("Model1", "Model2", [ops...], description=...)`; show the diff. |
| "…starting from Model2's fit" | add `seed_from="Model2"` to `derive`. |
| "ModelNg = ModelN with SOG thickness/roughness shared across energies (global fit)" | `p.derive("ModelN", "ModelNg", [{"op": "link", "layer": "SOG"}, ...])` (params default thickness+roughness; `"params": "thickness"` for one). Energy set is part of the model: `{"op": "link_energies", "energies": [...], "erange": [lo, hi]}` (default all data energies; resolved to data energies on save). `run fit` on a linked model = ONE global Sep-CMA-ES over its energy set (`--energies/--erange` refused); stored per energy under one run index with `global_chi2_total`; "best" = lowest global total. Report total χ², χ²/(N−P) and the shared values from the job output. Check first with `p.build_global("ModelNg")["checks"]`. Seeding from a per-energy model → shared start = median of the seeded values. |
| "…start only SOC/SiO2 from Model2's fit" | `derive(..., seed_from="Model2", seed_layers=["SOC", "SiO2"])` (instrument too unless `seed_instrument=False`); other layers start from recipe values. Use this when adding/splitting layers — a full seed would stack new layers on an already-full thickness. `derive` without `seed_from` drops the parent's seed. |
| "ModelNa = ModelN from its fit, widen the stuck SOG SLD bounds by 1" | `r, report = p.widen_stuck_sld("ModelN", "ModelNa", "SOG", step=1.0, tol_pct=10)` — seeds from the parent's fit; per energy, widens only the side where the fitted sld/isld ended within tol_pct % of the window (same test as the red "near bound" dots); writes bounds-only energy overrides. Print the report + `show_model`. Imag stuck at the 0 clamp is reported, not widened. |
| "change the bounds of X in ModelN" | `p.edit_model("ModelN", [ops])` — only if unfitted; if frozen, propose deriving a new model (optionally seeded from the fit). |
| "for 270 eV only start SOG at re=…, im=…" | op `set_energy_override` (see ops). "change 270 back" → `clear_energy_override`. |
| "show / list the models", "full model" | `show -p P ModelN` / `models -p P`; full detail = layer table + material sources + ops + per-energy starting SLDs from `build_objectives` + `free_parameter_table`. |
| "plot the loaded SLDs / optical constants" | `plot materials -p P [--models M \| --names A B]` |
| "fit models 1–3" | confirm, then `run fit -p P --models Model1 Model2 Model3 --gpu N --detach`; poll `status`. |
| "how did the fits go" | `status -p P`, `fits -p P [--per-energy]` |
| "bar graph / compare χ²" | `plot chi2 -p P --models ...` (default reduced χ²/N; `--metric raw` = notebook style) |
| "χ² vs energy" | `plot chi2-energy` |
| "compare reflectivity / fits" | `plot refl -p P --models ... [--energies ...\|--erange a b]` |
| "SLD of SOG vs energy for models 2,4 vs the Brewer6 reference" | `plot sld-energy -p P --layer SOG --models Model2 Model4 --ref <material-or-csv> ...` |
| "show ModelNa's new bounds on that SLD plot" | add `--bounds-from ModelNa` to `plot sld-energy` (works for unfitted models; widened energies marked ▲/▼). |
| "thickness of SOC vs energy" | `plot param -p P --param "SOC - thick" --models ...` |
| "depth profiles" | `plot profiles -p P --models ... [--energies ...] [--ref SOG SOC]` |
| "save/export Model3's SOG SLD (to seed the next sample)" | `export-sld -p P --model Model3 --layer SOG [--register-as SOG_B9M3]` |
| "fit the stoichiometry / density of SOG (from Model3's fit / a CSV)" | Register a search: `p.add_stoich(name=, source=, atoms=[...], counts=[...], density_range=(lo, hi), merge_points=, energy_mask=)` — source = material name \| `"Model3:SOG"` (fitted layer SLD) \| CSV path; counts entries int (fixed) \| `[lo, hi]` \| `[lo, hi, step]`. Fragment search: `mode="fragments", fragments=["SiO2", "CH3"], counts=[...]`. State the candidate count (`p.get_stoich(n).n_candidates()`, ~40 s per 1600 on 40 workers), then `stoich run NAME [NAME2] -p P [--workers N] --detach`; `stoich list` / `stoich show NAME`. Fix one count (or use a fixed fragment) — formulas differing by an overall factor have identical cost. Best density is refined within ±`refine_density` (default 0.1). |
| "stoichiometry plots" | `plot stoich-results --name N` (top-N bars + KK real/imag), `stoich-density` (RMSE vs ρ), `stoich-counts [--baseline C=40 ...]` (RMSE vs each count/multiplier, others at best), `stoich-overlay --name N [--sim FORMULA:RHO ...]` or `--source S --sim ...` (simulate without fitting). Results in `<project>/stoich/<name>/` (results.csv, best.json, best_kk_sld.csv = KK-consistent SLD of the best formula, registrable as a material). |
| "run JAXNS / NUTS on Model3 at 285–290" | confirm (time!), `run jaxns\|nuts -p P --model Model3 --erange 285 290 --gpu N --detach` |
| "uncertainty results / evidence" | `unc -p P --models ...`; plots `uncertainty`, `posterior`, `evidence`, `corner` |

## Model building API

```python
p = FitProject.open(PROJ)
p.new_model("Model1", [
    LayerSpec("air"),                                             # constant 0 SLD
    LayerSpec("SOG", "SOG", 335, 7, (330, 340, True), (4, 15, True), sld_offset=pm_offset(3)),
    LayerSpec("SOC", "SOC", 962, 4, (957, 967, True), (2, 20, True), sld_offset=pm_offset(0.1)),
    LayerSpec("SiO2", "SiO2", 17, 3, (15, 20, True), (1, 10, True)),  # no offset → SLD fixed
    LayerSpec("Si", "Si", 0, 5, (0, 0, False), (3, 7, False)),
], InstrumentSpec(scale=10, scale_bounds=(0.1, 50), bkg_bounds=(0.001, 1),
                  dq=0.7, dq_bounds=(0.5, 2.5)), description="...")
print(p.show_model("Model1"))
```
`LayerSpec(name, material, thickness, roughness, thickness_bounds, roughness_bounds,
sld_offset=None, sld_real_bounds=None, sld_imag_bounds=None)`; bounds are
`(lo, hi, vary)`. `pm_offset(re, im=None)` → `{"real": (-re, re, True), "imag": (-im, im, True)}`.

**Ops** (list of dicts for `derive` / `edit_model`):
- `{"op": "add_layer", "name": "SOG2", "above": "SOG" | "below": ..., "like": "SOG", "material": "SOG2", <LayerSpec fields>}` — `like` copies a layer then fields override.
- `{"op": "remove_layer", "name": L}`, `{"op": "move_layer", "name": L, "above"|"below": X}`, `{"op": "rename_layer", "old": A, "new": B}`
- `{"op": "set_layer", "layer": L, <fields>}` (value `None` clears a bound), `{"op": "set_material", "layer": L, "material": M}`
- `{"op": "set_instrument", <InstrumentSpec fields>}`
- `{"op": "set_energy_override", "layer": L, "energy": E | "energies": [...], "sld_real": .., "sld_imag": .., "sld_offset": .., "thickness": .., "roughness": ..}` — SLD bounds re-centre on the new start using the layer's offset.
  Give only `sld_offset` (no sld_real/sld_imag) to re-window the bounds around the tabulated SLD while keeping the current/seeded start (asymmetric OK, e.g. `{"real": (-3, 2, True)}`).
- `{"op": "clear_energy_override", "layer": L?, "energy": E?}`, `{"op": "seed_from_fit", "model": M}`

**Independent materials:** if the user wants a layer's optical constants tracked
separately (e.g. SOG2 from the same file as SOG), register a second material from
the same source (`src = p.material("SOG"); p.add_material("SOG2", kind=src.kind, path=src.path)`)
and `set_material`. Note: each layer's fit parameters are already independent;
only the starting table is shared otherwise — say so.

## Conventions (this user's data)

- Energies in **eV**; layer order **top → bottom** (air first, Si last).
- `sld_offset` = absolute offset (1e-6 Å⁻²) around the tabulated SLD at each
  energy; "±0.5" → `pm_offset(0.5)`; imag lower bound clamps at 0.
- `scale_bounds` / `bkg_bounds` are **multiplicative** on the start value;
  `bkg=None` = data minimum. **`bkg_bounds=(0.001, 1)` (bkg can only decrease)
  is intentional — do not flag it.** Typical ALS carbon settings: scale 9–10
  (data are not normalised; R ≈ 7 at low q), dq 0.7 fixed.
- Models named `Model1, Model2, …`; a second copy of a layer is `SOG2` etc.
- Fitted parameter names: `"<layer> - thick|rough|sld|isld"`, `scale`, `bkg`.
- Fits are per energy and independent; `normalize=False` for CMA-ES,
  `normalize=True` for NUTS/JAXNS (tempered posterior — report it with log Z).

## Protocol

1. **Before saving a model:** print `show_model` (layer table + diff vs parent)
   and list every value you assumed rather than were told (files, thicknesses,
   bounds, instrument). Save when the user's request was complete; otherwise
   ask. Sanity-check requested values against the tabulated SLD at that energy
   and mention big departures once.
2. **Frozen models:** a model with fits cannot be edited — derive a new one
   (offer `seed_from`).
3. **Before any `run`:** state models, energies, GPU, settings, rough time, and
   get a yes. Check `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
   and `status` to pick a free GPU; use `--detach` and poll `status`
   (or wait on the job json with a background loop). Rough times on this box:
   CMA-ES popsize 500, 4 energies ≈ 30 s (early stop); JAXNS 500 live points ≈
   30 min **per energy**; NUTS ≈ 2–3 min per energy. Warn before multi-hour jobs.
4. **Figures:** after every `plot`, `Read` the PNG to check it, then give the
   user a clickable markdown link to the file (they cannot see images you Read).
   Describe what it shows in 2–4 lines.
5. **Failures:** `status --job ID --tail 40`; job records keep the traceback.
6. Don't kill processes you didn't start; `pkill -f` patterns can match your own shell — kill by PID.

## Not implemented yet (milestone M6)

Global-fit JAXNS/NUTS (M6 step 2 — `run jaxns` on a linked model is not yet
global), linking SLD across energies, multi-edge stoichiometry (independent/stitched KK across edges),
Kramers–Kronig-constrained fitting, intensity-vs-energy scans. Say so and offer
to do it with the underlying Refltools functions directly.

## Tests

`$PY tests/test_harness.py` from `/homes/dfs1/Refltools` (plain asserts, ~3 min,
CPU). `RSOXR_SLOW=1` also runs real JAXNS on CPU (slow).
