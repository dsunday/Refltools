# Reflectometry model-building & fitting workflow

Reference for the pattern used to build refnx reflectometry models with `Refltools`,
fit them (CMA-ES point estimate → JAXNS nested sampling), compare structural
hypotheses via Bayesian evidence, and plot posterior uncertainty. This pattern is
used across the `LAMS*_MR.ipynb` notebooks in `Reflectivity/LAM/FittingNB/`.

## 0. Environment

Use the **`refnxlocal`** conda environment when working with this code (or similar
reflectivity-fitting codebases built on `Refltools`/refnx):

```bash
conda activate refnxlocal
```

## 1. Building a model (materials, layers, structure, objective)

Two calls build everything:

```python
materials, layers, structure, model_name = create_reflectometry_model(
    materials_list, layer_params, layer_order=layer_order,
    ignore_layers=['Si', 'air'], sample_name=sample_name, energy=energy)

model, objective = create_model_and_objective(
    structure=structure, data=Data, model_name=model_name,
    scale=1, bkg=None, dq=1,
    vary_scale=False, vary_bkg=False, vary_dq=False,
    scale_bounds=(0.5, 10), bkg_bounds=(0.01, 1.1), dq_bounds=(1.0, 20.0))
```

### `create_reflectometry_model` — `Model_Setup.py:32`

```python
create_reflectometry_model(materials_list, layer_params, layer_order=None,
                            ignore_layers=None, sample_name=None,
                            energy=None, wavelength=None, probe="x-ray")
```

- **`materials_list`** — list of material dicts, one of two approaches (auto-detected
  from the *first* entry; all entries must use the same one):
  - density approach: `{'name': 'SiO2', 'formula': 'SiO2', 'density': 2.65}` → built as
    a `MaterialSLD` (needs `energy` or `wavelength` to convert formula+density to SLD).
  - SLD approach: `{'name': 'X', 'real': ..., 'imag': ...}` → built as a plain `SLD`.
- **`layer_order`** — list of layer names, **top to bottom** (e.g.
  `['air', 'MOXA', 'UL', 'SiO2', 'Si']`). The `Structure` is assembled by chaining
  `Layer[layer_order[0]] | Layer[layer_order[1]] | ...` (refnx's `|` operator).
- **`ignore_layers`** — layer names excluded from the auto-generated `model_name`
  (default `["Si", "SiO2"]`). Purely cosmetic/labeling — does **not** affect what's
  fit or how the structure is built.
- Returns `(materials, layers, structure, model_name)`.

**Gotcha:** every name in `layer_order` (and every name you put bounds on in
`layer_params`) must also appear in `materials_list`. A layer referenced in
`layer_order`/`layer_params` but missing from `materials_list` causes a `KeyError`
when the structure is assembled (`Layer[lname]` lookup fails silently until then —
the material-building loop just skips layers that aren't in `materials_list`, so the
error only surfaces at structure-assembly time). This is a real bug that was hit
building a 2-layer variant that added a new material (`MOXB`) to `layer_order` and
`layer_params` but forgot to add it to `materials_list`.

### `create_model_and_objective` — `Model_Setup.py:313`

```python
create_model_and_objective(structure, data, model_name=None, scale=1.0,
                            bkg=None, dq=1.6, q_offset=0.0,
                            vary_scale=True, vary_bkg=True,
                            vary_dq=False, vary_qoffset=False,
                            scale_bounds=(0.1, 10), bkg_bounds=(0.01, 10),
                            dq_bounds=(0.5, 2.0), qoffset_bounds=(-0.01, 0.01),
                            transform='logY')
```

Returns `(model, objective)` — a refnx `ReflectModel` and refnx `Objective`.
`model(q_array)` evaluates reflectivity; `objective.chisqr()` gives the current fit
quality; `model.structure` / `structure.sld_profile()` gives the depth-SLD profile.

## 2. Parameter formatting

`layer_params[name]` is a dict of optional keys:

| Key | Meaning |
|---|---|
| `thickness` / `roughness` | initial values (Å) |
| `thickness_bounds` / `roughness_bounds` | `(lower, upper, vary: bool)` |
| `density_bounds` | `(lower, upper, vary: bool)` — density approach only |
| `sld_real_bounds` / `sld_imag_bounds` | `(lower, upper, vary: bool)` — SLD approach only |

Example:

```python
'SiO2': {'thickness': 10, 'roughness': 8,
         'thickness_bounds': (5, 30, False),      # fixed thickness
         'roughness_bounds': (0.1, 10, True),      # free
         'density_bounds': (2.0, 2.65, True)},     # free
```

**Gotcha:** for density-approach materials, the layer's *initial* density comes from
`materials_list`'s `'density'` field — **not** from any `'density'` key you put inside
`layer_params`. A `'density'` entry inside `layer_params` is inert/documentation-only;
only `density_bounds` (lower/upper/vary) actually does anything there. If you want a
layer's initial density to differ from what's in `materials_list`, you have to edit
`materials_list` itself.

### Reading/writing free parameters — `gpu_reflect.py:445,486`

```python
extract_free_params(objective, excluded_name=None)   # -> list[Parameter]
set_free_params(objective, values, excluded_name=None)  # sets in-place
```

> `extract_free_params`: "Return free Parameter objects from objective, optionally
> excluding one by name. Returns them in the same traversal order as
> `objective.parameters.flattened()`, which is the order used by `objective.setp()`."
>
> `set_free_params`: "Set the free parameters of objective (excluding
> `excluded_name`) to values. Modifies the objective in-place. `values` must match
> the order from `extract_free_params(objective, excluded_name)`."

Both calls use the same underlying order, so it's safe to round-trip a flat
`np.ndarray` (e.g. one row of posterior samples) through `set_free_params` without
separately tracking parameter names.

## 3. Optimization workflow

**Step 1 — CMA-ES point estimate** (`batch.py:1460`):

```python
cmaes_1L = batch_fit_selected_models_cmaes(
    objectives_dict={energy: obj}, energy_list=[energy],
    popsize=20, n_generations=1000, tol=1e-4, patience=5,
    model_name='1L', sample_name=sample_name, h5_filepath=H5_file, verbose=True)
obj = cmaes_1L['fitted_objectives'][energy]
```

Signature: `batch_fit_selected_models_cmaes(objectives_dict, energy_list=None,
popsize=20, n_generations=500, seed=0, verbose=True, tol=1e-4, patience=5,
check_every=10, sample_name=None, model_name=None, h5_filepath=None,
run_index=None, normalize=False)`.

- Sep-CMA-ES (diagonal-covariance CMA-ES) on GPU, `popsize` candidates/generation.
- Early stopping: halts once the relative improvement in mean best χ² is below
  `tol` for `patience` consecutive checks (checked every `check_every` generations).
- Returns a dict: `'fitted_objectives'` ({energy: Objective at best-fit params}),
  plus `'individual_results'`, `'summary_stats'`, `'fitted_energies'`,
  `'elapsed_sec'`, `'generations_run'`, `'converged'`.
- H5 storage is namespaced `/{sample_name}/{energy}/{model_name}/run_N/`
  (`h5io.py:282`, auto-incrementing `run_N`) — **multiple samples and multiple
  models can safely share one `h5_filepath`.** This is *not* true of raw `.npz`
  caching — see the gotcha below.

**Step 2 — JAXNS nested sampling** (`gpu_nested_sampler.py:258`):

```python
ns_1L = run_nested_sampling(
    obj, max_samples=100_000, num_live_points=500, seed=0,
    n_posterior_samples=2000, verbose=True, normalize=True)
```

`normalize=True` divides the log-likelihood by the number of data points before
sampling — the standing convention in these notebooks, so evidence values are
comparable across datasets/models with different numbers of Q points. Returns a
`NestedSamplingResult`: `log_Z_mean`/`log_Z_std` (evidence), `ESS`, `H_mean`
(information gain, nats), `samples` (equal-weight posterior draws, shape
`(n_posterior_samples, n_free)`), `posterior_mean`/`std`/`median`, `param_names`,
`run_seconds`.

**Step 3 — evidence comparison** (`gpu_nested_sampler.py:648`):

```python
compare_evidence({'With SiO2': ns_1L, 'No SiO2': ns_1L_noSiO2})
```

Prints a log Z table, pairwise Bayes factors, and a Jeffreys'-scale interpretation
of which hypothesis the evidence favors.

**Caching pattern** used throughout — a `force_rerun` flag gates whether a step
re-executes or loads a previous result:

```python
def _npz_done(fname):
    return os.path.exists(save_dir + fname)

def save_cmaes_params(path, objective): ...   # param_names, param_values
def save_jaxns(path, result): ...              # full NestedSamplingResult fields
def load_jaxns(path): ...                      # -> SimpleNamespace w/ same fields
```

> ⚠️ **Gotcha (hit in practice, not hypothetical)**: `.npz` cache filenames **must**
> be namespaced by `sample_name`, e.g. `f'{sample_name}_jaxns_1L.npz'`, not bare
> `'jaxns_1L.npz'`. Multiple sample notebooks sharing one `save_dir` with bare
> filenames will silently overwrite each other's cached CMA-ES/JAXNS results —
> whichever notebook ran most recently wins. A later "reload from disk" cell (e.g.
> in the uncertainty-plotting section) then silently loads a *different sample's*
> posterior and applies it to the current sample's data, producing a plausible-looking
> but wrong uncertainty envelope with no error raised. This actually happened across
> the LAMS3–LAMS7 notebooks and was fixed by prefixing every `.npz` filename with
> `sample_name`; keep doing that for any new sample notebook.

## 4. Model-comparison pattern (with/without a layer, 1-layer vs 2-layer)

To test a structural hypothesis (does this layer exist? does it need to be split
into two?), build a sibling model variant rather than mutating the original:

- **Remove a layer**: copy `materials_list`, `layer_order`, and `layer_params`,
  drop the layer's entries from all three, adjust `ignore_layers` accordingly.
- **Split a layer into two**: copy the three structures, replace one material
  entry (e.g. `MOXA`) with two (`MOXA` + `MOXB`), give each its own
  thickness/roughness/bounds, and add both to `layer_order`.

Give each variant fully distinct variable names (`obj_noSiO2`, `obj2`,
`structure_2L_noSiO2`, ...) rather than reusing generic `model`/`structure` names —
reusing generic names across variants makes it easy for a later cell to silently
reference the wrong model's `model`/`structure` since they get overwritten each time
a new variant is fit. Fit each variant independently through the full CMA-ES → JAXNS
pipeline above, then run `compare_evidence` across all variants in one dict to see
which structure the data actually supports.

## 5. JAXNS posterior uncertainty plotting

Pattern for turning nested-sampling posterior draws into credible-interval bands for
both the reflectivity curve and the SLD depth profile (adapted from
`Mixed12_JAX_V3_Plot.ipynb` in the UC_Peptoid project, simplified to JAXNS-only
since these notebooks have no separate MCMC/NUTS results to plot):

```python
def plot_envelopes(samples, obj, model, structure_obj, label, color='C0'):
    orig_vals = np.array([p.value for p in extract_free_params(obj)])

    idx = rng.choice(len(samples), min(N_plot, len(samples)), replace=False)
    R_curves, SLD_curves = [], []
    for s in samples[idx]:
        set_free_params(obj, s)
        R_curves.append(model(q_plot))
        z_i, sld_i = structure_obj.sld_profile()
        SLD_curves.append(sld_i.real)

    set_free_params(obj, orig_vals)   # restore

    R_arr, SLD_arr = np.array(R_curves), np.array(SLD_curves)
    # np.percentile(R_arr, 2.5/97.5, axis=0) + np.median(R_arr, axis=0) -> R(q) band
    # same for SLD_arr vs z_i -> SLD(z) band
```

Draw `N_plot` (e.g. 300) random rows from `NestedSamplingResult.samples`, evaluate
the model/structure at each, stack into arrays, and take percentiles across the
sample axis. Always restore the objective's original parameter values afterward so
the plotting call has no side effect on `obj`.

**Note:** `structure.sld_profile()` called with `z=None` always returns a
fixed-length 500-point `z` grid (refnx `structure.py:508`), even though the *range*
of that grid shifts with total thickness across posterior draws. That means
stacking `SLD_curves` from different draws into one `np.array` is always shape-safe
— no need to interpolate onto a common grid first.
