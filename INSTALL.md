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
- Windows: see [section 7](#7-windows).

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

## 7. Windows

There are two routes:

**WSL2 (recommended, and the only way to use a GPU on Windows).** Install WSL2 with
Ubuntu (`wsl --install` in an admin PowerShell) and the normal NVIDIA Windows
driver. Then follow sections 2–6 *inside Ubuntu* exactly as written, using
`environment.yml`. Keep data inside the WSL filesystem (e.g. `~/data`); access
to `/mnt/c/...` is much slower.

**Native Windows (CPU only).** Use the Miniforge Prompt (or PowerShell after
`conda init`), and:

1. Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   with the **"Desktop development with C++"** workload. This is needed to compile refnx.
2. Create the env from the Windows file:
   ```bat
   git clone https://github.com/dsunday/Refltools.git
   cd Refltools
   mamba env create -f environment-windows.yml
   mamba activate refltools
   ```
   This uses CPU-only `jax` and sets `PYTHONUTF8=1`, so files with σ, χ², ° are
   read and written as UTF-8.
3. Install the refnx fork and verify exactly as in sections 3 and 5.

Differences from Linux:
- Cancelling a job (`python -m rsoxr_harness cancel ...`) force-kills the worker
  and its child processes. The job is marked `cancelled`, but the worker gets no
  chance to tidy up, so a partly written output may be left behind.
- Job and stoichiometry locks are exclusive on Windows (readers wait for each
  other). This is slower in theory, but not noticeable in practice.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| refnx build fails (compiler / meson errors) | Env not active, or created without the build tools. Re-create it from `environment.yml`. |
| `ModuleNotFoundError: refnx.reflect._jax_reflect` | Upstream refnx is installed. Run `pip uninstall refnx`, then repeat step 3. |
| `jax.devices()` shows only CPU on a GPU machine | Driver too old (`nvidia-smi`), the CPU-only jax line was used, or a system CUDA on `LD_LIBRARY_PATH` is shadowing the wheels (`unset LD_LIBRARY_PATH`). |
| Windows: refnx build can't find `cl.exe` / MSVC | Visual Studio 2022 Build Tools with the C++ workload aren't installed. Install them, open a new prompt, re-activate the env, and repeat step 3. |
| Windows: `UnicodeEncodeError` / `UnicodeDecodeError` | Env created from `environment.yml` instead of `environment-windows.yml`. Run `conda env config vars set PYTHONUTF8=1`, then re-activate. |
| CUDA out-of-memory with several jobs | Set `CUDA_VISIBLE_DEVICES` per job, and keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`. |
