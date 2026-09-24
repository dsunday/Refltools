"""
materials.py
------------
Where a material's tabulated SLD(E) comes from, and how to turn that into the
[Energy_eV, Real_SLD, Imag_SLD] array the batch pipeline interpolates.

kinds
    file     – text/CSV table (np.loadtxt whitespace, or CSV with a header as
               written by Model_Setup.save_material_sld / analysis.export_sld_csv)
    fit_csv  – same loader; flags the table as a previous fit result
    formula  – refnx MaterialSLD from formula + density (g/cm³), eV grid
"""

import os
from dataclasses import dataclass, field, asdict

import numpy as np

KINDS = ("file", "fit_csv", "formula")


@dataclass
class MaterialSource:
    kind: str
    path: str | None = None
    formula: str | None = None
    density: float | None = None
    description: str = ""
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"material kind '{self.kind}' not in {KINDS}")
        if self.kind in ("file", "fit_csv") and not self.path:
            raise ValueError(f"kind='{self.kind}' needs path=")
        if self.kind == "formula" and (not self.formula or self.density is None):
            raise ValueError("kind='formula' needs formula= and density=")

    def to_json(self):
        return {k: v for k, v in asdict(self).items() if v not in (None, "", {})}

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def label(self):
        if self.kind == "formula":
            return f"{self.formula} @ {self.density:g} g/cm³"
        return os.path.basename(self.path)


def load_table(path):
    """Load an SLD table; returns ndarray with ≥3 columns [E, Re, Im, ...]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as fh:
        first = fh.readline()
    delim = "," if "," in first else None
    try:
        float(first.replace(",", " ").split()[0])
        skip = 0
    except ValueError:
        skip = 1
    arr = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: need ≥3 columns (E, Re, Im), got {arr.shape[1]}")
    return arr[np.argsort(arr[:, 0])]


def resolve(src, energies_eV, project_dir="."):
    """MaterialSource → ndarray [E_eV, Re, Im]."""
    if src.kind in ("file", "fit_csv"):
        path = src.path if os.path.isabs(src.path) else os.path.join(project_dir, src.path)
        return load_table(path)[:, :3]
    # formula: tabulate on the requested grid (plus a margin so interpolation
    # never extrapolates at the ends)
    from Model_Setup import generate_sld_array_from_material
    e = np.unique(np.concatenate([np.asarray(energies_eV, float),
                                  [min(energies_eV) - 1, max(energies_eV) + 1]]))
    return generate_sld_array_from_material(src.formula, src.density, e,
                                            probe="x-ray")


def interp_at(arr, energy):
    """(real, imag) of a tabulated array at one energy (linear)."""
    return (float(np.interp(energy, arr[:, 0], arr[:, 1])),
            float(np.interp(energy, arr[:, 0], arr[:, 2])))
