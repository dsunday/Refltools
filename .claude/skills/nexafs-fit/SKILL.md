---
name: nexafs-fit
description: DRAFT. Peak-fit NEXAFS spectra with Refltools (Gaussian peaks + erf/decay edge, NEXAFS_BuildingBlock) — LARIAT surface NEXAFS (processed CSVs) and RSoXR-fitted optical constants (isld from SLD_Fits CSVs) for Brewer SOC/SOG/resist layers; seed-model library, relaxed refits, frozen-peak isld fits, energy-shift scan, summary tables. Use whenever the user talks about NEXAFS peak fitting, peak tables, isld/beta peak decomposition, LARIAT spectra, or section G of the Brewer to-do.
---

# NEXAFS peak fitting (Refltools/nexafs_batch.py)  — DRAFT, pilot only

Env: `PY=/homes/dfs1/mambaforge/envs/refnxlocal/bin/python`; `cd /homes/dfs1/Refltools`. Test: `$PY tests/test_nexafs_batch.py` (no pytest in this env).

## Pieces
- `nexafs_models.py` — seed/bound models extracted from `LARIAT/BrewerNEXAFS_Fits.ipynb`: `SOC_CARBON_BAKE`, `SOC_CARBON_UV`, `SOG_CARBON_BAKE` (9 peaks each, C K-edge), `SOG_SI_*` (Si K). Each = `dict(peaks, edge, baseline)`; bounds are `(lo, hi, fit_flag)`; edge params only vary if `<name>_bounds` is given.
- `nexafs_batch.py` — `load_source`, `fit_one`, `relaxed`, `fit_isld`, `normalise_isld`, `save_fit`, `run_pilot`. Engine: `NEXAFS_BuildingBlock.fit_nexafs_spectrum` (curve_fit, trf).
- Outputs: `/homes/dfs1/Reflectivity/Brewer/Nov2025/NEXAFS_Fits/<BrewerN_sample>/{lariat,isld}/{peaks.csv,summary.csv,result.pkl,fit.png}`.

## Data
- LARIAT: `/homes/dfs1/LARIAT/Brewer_Spectra/March2026/FallBatch/Brewer<N>_<sample>[_Carbon]_processed.csv` (Energy, Processed_Intensity; pre-edge 0, post-edge ~1.43). Carbon filename has no suffix for SOG UV / Bake refs (e.g. `Brewer3_SOG_UV200_processed.csv`). Brewer13 files are labelled Brewer12_NTD_UV_Developed (unverified).
- isld: `Reflectivity/Brewer/Nov2025/SLD_Fits/BrewerN/*.csv` (Energy, SLD_Real, SLD_Imag; ~30–40 irregular points, C edge). Beware SOG vs SOG2 name matching.

## Conventions / lessons (pilot: Brewer1 SOC UV, Brewer3 SOG UV200)
1. Notebook seeds are sample-specific (e.g. peak-2 height frozen at 1.7/2.6). Used as-is they fit Brewer1 (R²=0.983) but not Brewer3 (R²=0.64). Use `relaxed(model)` (free π* peak, edge height/decay, height cap 6) → R² 0.998 / 0.996. Always report seed-as-is and relaxed R².
2. Check `*_at_bound` columns in peaks.csv; Brewer3 relaxed fit has peak 2 energy and peaks 6/8 at bounds and a peak1/peak2 degeneracy near 285 eV.
3. isld: normalise (`normalise_isld`: pre-edge mean E≤282 → 0, post-edge E≥320 → edge height), freeze peak energies/widths from the LARIAT fit of the same sample, fit heights only, scan an energy shift (E_nexafs = E_rsoxr + shift, ±2 eV, 0.25 steps). Best shift was −1.25 eV for both pilot samples (compare the notebook's 1.2 eV offset; sign convention differs — confirm).
4. Tail anchoring (`fit_isld(tail_decay=...)`): fixes edge decay and solves edge height through the >=320 eV points. Decide case by case from the data (`isld_coverage`; the `TAIL_DECAY` table in nexafs_batch.py), not by default — use it when the 300–310 eV gap leaves broad peaks undefined and the high-energy points are missed. Brewer1 SOC UV: decay 0.005 (0.003–0.0075 tried; the LARIAT decay 0.0136 drives peak 9 to 0 and R² to 0.92). Otherwise edge height/decay float freely (Brewer3).
5. isld grids have almost no points above ~300 eV → the broad σ* peaks (7–9) are unconstrained; do not interpret their heights.
6. The notebook's fit cells reference undefined variables (`*_day2`, `*_Sept2025` vs `*_March2025`); don't run it top to bottom, use nexafs_models.py.

## Not implemented yet
Oxygen/Si edges in the batch driver, SOG2/resist model seeds, all-sample manifest (filename irregularities), to-do ticking via make_todo.py, harness `rsoxr_harness nexafs` integration (alternative engine, not wrapped).

## Co-fit (nexafs_cofit.py) — SOG baseline decided 2026-10-30
`python nexafs_cofit.py sog|sog_wide|sog_variants|soc` → outputs in `Nov2025/NEXAFS_Fits/CoFit/`.
Shared peak energies/widths (free within bounds), per-sample heights + edge height/decay; errors from the full covariance (delta method) incl. sums/ratios; LARIAT errors are statistical only.
- **SOG baseline model (use for SOG-vs-SOG comparisons): wide bounds, WITHOUT the 291.5 eV peak** (`run(names, kind, SOG_CARBON_BAKE, tag, wide=True, drop=(6,))`). wide = energy bounds ±0.5 eV, widths ×0.5/×1.6 beyond seed bounds. Seed bounds distorted the fit (LARIAT RSS 6.6→2.6, isld 13.9→8.8; peak 1 FWHM 0.24, peak 7 ~2).
- Peak 6 (291.5): only supported in Brewer3 LARIAT (ratio to 288.8 = 0.42±0.17); isld AIC/BIC prefer dropping it; dropping it collapses peak 8 (297 eV) to zero height.
- **Open issue:** the 302.7 eV peak is unphysically large (FWHM ~10.4 eV); revisit its bounds/model later.
- SOC test (Brewer1/2, all peaks kept; SOC peak 6 is ~290.4 eV): wide bounds also help (LARIAT RSS 0.87→0.59, 8→1 params at bound; isld 1.43→1.08). isld SOC co-fit has only 2 samples/75 points for 40 params — overfit risk; Brewer1 isld edge decay still goes to 0 (needs tail anchoring, not applied in co-fit).
