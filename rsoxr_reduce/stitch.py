"""
stitch.py
---------
Pure-function reduction math, numerically identical to the widget path
``RSoXRProcessor.reduce_data_with_open_beam_option`` (trim -> background ->
÷ I0 -> drop non-positive -> cumulative median-ratio stitch -> Q, sorted).

Trim convention (legacy): ``(start, end)`` keeps ``data[start : len+end]`` when
``end < 0`` (so ``end = -1`` drops the last point) or ``data[start:end]``.
"""

import numpy as np

HC = 12399.0          # eV·Å, as RSoXRReduction (1239.9 nm·eV × 10)


def apply_trim(data, start, end):
    end_idx = len(data) + end if end < 0 else end
    return data[start:end_idx, 0:2].copy()


def kept_indices(n, start, end):
    end_idx = n + end if end < 0 else end
    return np.arange(start, max(start, end_idx))


def to_q(theta_deg, energy):
    return 4 * np.pi * np.sin(np.radians(theta_deg)) / (HC / energy)


def median_ratio(combined, scan):
    """Legacy scale factor of ``scan`` onto ``combined`` (both (θ, y)).

    Returns (scale, ratios) where ratios are combined/scan for every scan
    point inside the overlap (nearest-angle pairing).  scale = 1.0 and an
    empty list when there is no overlap.
    """
    prev_max = combined[:, 0].max()
    curr_min = scan[:, 0].min()
    ratios = []
    if curr_min <= prev_max:
        oc = combined[combined[:, 0] >= curr_min]
        os_ = scan[scan[:, 0] <= prev_max]
        for ang in os_[:, 0]:
            ic = np.argmin(np.abs(oc[:, 0] - ang))
            is_ = np.argmin(np.abs(os_[:, 0] - ang))
            if oc[ic, 1] > 0 and os_[is_, 1] > 0:
                ratios.append(oc[ic, 1] / os_[is_, 1])
    if ratios:
        return float(np.median(ratios)), ratios
    return 1.0, ratios


def reduce_sections(scans, trims, backgrounds, energy, i0=None,
                    remove_zeros=True, error_frac=0.01, scale_overrides=None):
    """Stitch sections into one curve.

    scans: list of raw arrays (N×≥2), stitch order.  trims: [(start, end)].
    backgrounds: per-section constant.  i0: open-beam intensity at ``energy``
    (None -> no I0 division, curve normalised to its max like the legacy code).
    scale_overrides: optional {section_index: scale}.

    Returns dict(q_data (N×3 sorted by Q), theta_data (N×2 unsorted, stitched
    order), sections=[per-section dict(theta, y, scale, n_overlap, ratios)]).
    """
    scale_overrides = scale_overrides or {}
    per = []
    refl = bc_refl = None
    for i, (data, (s, e), bg) in enumerate(zip(scans, trims, backgrounds)):
        seg = apply_trim(data, s, e)
        if bg != 0.0:
            seg[:, 1] = seg[:, 1] - bg
        bc = seg.copy()
        if i0 is not None:
            seg[:, 1] = seg[:, 1] / i0
        n_drop = 0
        if remove_zeros:
            m = seg[:, 1] > 0
            n_drop = int((~m).sum())
            seg = seg[m]
            bc = bc[bc[:, 1] > 0]
        info = {"n_kept": int(len(seg)), "n_nonpositive": n_drop,
                "scale": 1.0, "ratios": [], "n_overlap": 0}
        if i == 0:
            refl, bc_refl = seg, bc
        else:
            scale, ratios = median_ratio(bc_refl, bc)
            if i in scale_overrides:
                scale = float(scale_overrides[i])
                info["override"] = True
            info.update(scale=scale, ratios=ratios, n_overlap=len(ratios))
            seg[:, 1] *= scale
            bc[:, 1] *= scale
            refl = np.concatenate((refl, seg))
            bc_refl = np.concatenate((bc_refl, bc))
        info["theta"] = seg[:, 0].copy()
        info["y"] = seg[:, 1].copy()
        per.append(info)
    if i0 is None:
        refl[:, 1] = refl[:, 1] / np.max(refl[:, 1])
    q = to_q(refl[:, 0], energy)
    out = np.zeros((len(q), 3))
    out[:, 0], out[:, 1], out[:, 2] = q, refl[:, 1], refl[:, 1] * error_frac
    out = out[out[:, 0].argsort()]
    return {"q_data": out, "theta_data": refl, "sections": per}


def write_dat(path, q_data):
    """Same header/format as the widget output (np.savetxt default fmt)."""
    np.savetxt(path, q_data, delimiter="\t", header="Q(Å⁻¹)\tIntensity\tError",
               comments="")


# ----------------------------------------------------------------------
# open beam
# ----------------------------------------------------------------------
class OpenBeam:
    """Open-beam I0(E), linear interpolation with extrapolation (legacy)."""

    def __init__(self, path):
        d = np.loadtxt(path, skiprows=1, ndmin=2)
        o = np.argsort(d[:, 0])
        self.path = path
        self.energy = d[o, 0]
        self.intensity = d[o, 1]

    @property
    def erange(self):
        return float(self.energy.min()), float(self.energy.max())

    def covers(self, e):
        lo, hi = self.erange
        return lo <= e <= hi

    def __call__(self, e):
        from scipy.interpolate import interp1d
        f = interp1d(self.energy, self.intensity, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        return float(f(e))

    def at(self, energies):
        return np.interp(energies, self.energy, self.intensity)
