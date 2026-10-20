# Installing Refltools (RSoXR fitting and reduction harnesses)

Refltools runs on top of a **fork of refnx** (`github.com/dsunday/refnx`, branch
`JAX`). The fork adds the JAX reflectivity kernels (`refnx.reflect._jax_reflect`)
used by the GPU fitters, NUTS (BlackJAX) and nested sampling (JAXNS). The
upstream `pip install refnx` / `conda install refnx` packages **will not work**.

## 1. Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) (provides `mamba`/`conda`) and `git`.
- For GPU use: Linux with an NVIDIA GPU and driver ≥ 525 (`nvidia-smi`). You do
  **not** need a system CUDA toolkit; the JAX wheels ship their own CUDA 12 libraries.
- CPU-only machines (including macOS) work too, just slower for CMA-ES, NUTS and JAXNS.

## 2. Get Refltools and create the environment

```bash
git clone https://github.com/dsunday/Refltools.git
cd Refltools
mamba env create -f environment.yml      # creates the env "refltools"
mamba activate refltools
```

**CPU-only:** before `mamba env create`, edit `environment.yml` and change
`jax[cuda12]==0.10.0` to `jax==0.10.0`.

## 3. Install the refnx fork

With the `refltools` env active:

```bash
pip install --no-deps --no-build-isolation "git+https://github.com/dsunday/refnx.git@JAX"
```

- `--no-build-isolation` compiles refnx against the env's numpy, using the
  compilers/meson/Cython already in the env.
- `--no-deps` skips refnx's GUI dependencies (PyQt6 etc.). The runtime
  dependencies are already in `environment.yml`.

If you also want to edit refnx, use an editable install from a clone instead:

```bash
git clone -b JAX https://github.com/dsunday/refnx.git
cd refnx
pip install --no-deps --no-build-isolation -e .
```

To update later, re-run the same `pip install` command (or `git pull` in the
clone; editable installs rebuild on import).

## 4. Optional extras

| For | Install |
|---|---|
| TFP-based paths in `gpu_nested_sampler.py` | `pip install tfp-nightly` |
| PyMC path in `Model_Setup.py` | `mamba install -c conda-forge pymc` |

The harnesses don't need either one.

## 5. Check the install

From the `Refltools` folder:

```bash
python -c "import jax, refnx.reflect._jax_reflect; print(jax.devices())"
python -m rsoxr_harness --help
python tests/test_harness.py           # fitting harness, should end "N/N passed"
python tests/test_reduce_harness.py    # reduction harness
```

On a GPU machine `jax.devices()` should list `CudaDevice(...)` entries. If you
see only `CpuDevice`, check Troubleshooting below.

## 6. Using it

- Refltools is a flat folder of modules, **not** an installed package. Run the
  harnesses from the repo root:
  ```bash
  python -m rsoxr_harness ...    # fitting
  python -m rsoxr_reduce  ...    # raw-data reduction
  ```
  In notebooks or scripts elsewhere, add the folder to the path first:
  `sys.path.append('/path/to/Refltools')`, or set `PYTHONPATH`.
- GPUs: the harness sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` itself so JAX
  doesn't grab the whole card. On shared machines, pick a GPU with
  `CUDA_VISIBLE_DEVICES=<n>`.
- To use the Jupyter env: `python -m ipykernel install --user --name refltools`.
- **Claude Code:** open the `Refltools` folder as the project. The
  `rsoxr-fit` and `rsoxr-reduce` skills in `.claude/skills/` then load
  automatically and drive the harnesses from plain-text requests.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| refnx build fails (compiler / meson errors) | Env not active, or created without the build tools. Re-create it from `environment.yml`. |
| `ModuleNotFoundError: refnx.reflect._jax_reflect` | Upstream refnx is installed. Run `pip uninstall refnx`, then repeat step 3. |
| `jax.devices()` shows only CPU on a GPU machine | Driver too old (`nvidia-smi`), the CPU-only jax line was used, or a system CUDA on `LD_LIBRARY_PATH` is shadowing the wheels (`unset LD_LIBRARY_PATH`). |
| CUDA out-of-memory with several jobs | Set `CUDA_VISIBLE_DEVICES` per job, and keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`. |
