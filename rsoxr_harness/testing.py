"""
testing.py
----------
Synthetic fixture: a small air / Film / SiO2 / Si sample simulated at a few
carbon-edge energies, written in the same file layout the real pipeline
loads, plus a FitProject pointing at it.
"""

import os

import numpy as np

from . import recipe as R
from .project import FitProject

TRUE = {"Film": (250.0, 5.0), "SiO2": (15.0, 3.0)}   # thickness, roughness


def _film_table(energies):
    e = np.linspace(min(energies) - 10, max(energies) + 10, 81)
    re = 12.0 - 4.0 * np.exp(-((e - 285.0) / 2.0) ** 2)
    im = 1.0 + 3.0 * np.exp(-((e - 286.0) / 1.5) ** 2)
    return np.column_stack([e, re, im])


def true_recipe():
    return R.ModelRecipe(
        name="Truth",
        layers=[
            R.LayerSpec("air"),
            R.LayerSpec("Film", "Film", *TRUE["Film"],
                        thickness_bounds=(230, 270, True),
                        roughness_bounds=(1, 10, True),
                        sld_offset=R.pm_offset(2.0)),
            R.LayerSpec("SiO2", "SiO2", *TRUE["SiO2"],
                        thickness_bounds=(10, 20, True),
                        roughness_bounds=(1, 6, True)),
            R.LayerSpec("Si", "Si", 0, 4.0),
        ],
        instrument=R.InstrumentSpec(scale=1.0, bkg=1e-6, vary_bkg=False, dq=1.0))


def make_synthetic_project(tmpdir, energies=(280.0, 285.0, 290.0),
                           noise=0.03, seed=0):
    """Write data + SLD tables under tmpdir and return an initialised project."""
    from batch import simulate_reflectivity_profiles

    rng = np.random.default_rng(seed)
    data_dir = os.path.join(tmpdir, "data")
    oc_dir = os.path.join(tmpdir, "oc")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(oc_dir, exist_ok=True)

    tables = {"Film": _film_table(energies)}
    refl = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables["Si"] = np.loadtxt(os.path.join(refl, "OC", "Si_SLD.txt"))[:, :3]
    tables["SiO2"] = np.loadtxt(os.path.join(refl, "OC", "SiO2_Dean_SLD.txt"))[:, :3]
    paths = {}
    for n, arr in tables.items():
        paths[n] = os.path.join(oc_dir, f"{n}_SLD.csv")
        np.savetxt(paths[n], arr, delimiter=",", header="Energy,SLD_Real,SLD_Imag",
                   comments="")

    truth = true_recipe()
    kw = truth.to_generate_kwargs(tables)
    q = np.linspace(0.01, 0.25, 150)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        sim, _, _ = simulate_reflectivity_profiles(
            energy_list=list(energies), q_values=q, sample_name="Synth",
            dq=1.0, scale=1.0, bkg=1e-6,
            **{k: kw[k] for k in ("material_sld_arrays", "constant_materials",
                                   "base_layer_params", "layer_order")})
    for e, d in sim.items():
        r = d["reflectivity"]
        dr = noise * r
        r_noisy = r + rng.normal(0, dr)
        np.savetxt(os.path.join(data_dir, f"Synth_{e:.1f}eV.dat"),
                   np.column_stack([q, np.abs(r_noisy), dr]))

    proj = FitProject.create(os.path.join(tmpdir, "project"), "Synth",
                             data_dir, file_type="all")
    for n, p in paths.items():
        proj.add_material(n, kind="file", path=p)
    return proj


def base_recipe(name="Model1"):
    """Starting guess recipe for the synthetic project (offset from truth)."""
    r = true_recipe()
    r = R.set_layer(r, "Film", thickness=245.0, roughness=4.0)
    r = R.set_layer(r, "SiO2", thickness=14.0)
    r = R.set_instrument(r, bkg=None, vary_bkg=True, bkg_bounds=(0.1, 10.0),
                         scale_bounds=(0.5, 2.0))
    r.name = name
    r.description = "synthetic baseline"
    return r
