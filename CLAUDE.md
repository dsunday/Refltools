# Refltools

Flat folder of Python modules for soft X-ray reflectivity (RSoXR) reduction and
fitting on top of the local refnx fork (`/homes/dfs1/refnx`, branch `JAX`).
Not a package: notebooks do `sys.path.append('/homes/dfs1/Refltools')` and import
modules directly (`batch`, `h5io`, `Model_Setup`, `gpu_*`, `Plotting_Refl`, ...).

- Environment: `/homes/dfs1/mambaforge/envs/refnxlocal/bin/python` (editable refnx,
  jax+CUDA, blackjax, jaxns, h5py, corner). No pytest there.
- Model-building/fitting guide: `MODEL_GUIDE.md`.
- **Plain-text fitting harness:** `rsoxr_harness/` (CLI `python -m rsoxr_harness`),
  driven by the `rsoxr-fit` skill in `.claude/skills/rsoxr-fit/SKILL.md`. Use it
  for any fitting request instead of writing notebook-style code.
- Tests: `python tests/test_harness.py` (harness only).
- GPUs are shared: check `nvidia-smi` first, set `CUDA_VISIBLE_DEVICES` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the harness job runner does this.
- Known issue (not fixed): `h5io.extract_sld_from_h5(materials_filter=...)`
  matches by substring ("SOG" also returns "SOG2").
