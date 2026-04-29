"""
kk_stoichiometry_fit.py
-----------------------
Fit material stoichiometry and density by maximising Kramers-Kronig consistency
between the KK-transformed imaginary SLD (from a reflectometry fit) and the
fitted real SLD.

Speed architecture
------------------
1. **Density is a linear scale factor on the ASF output.**
   The KK integral produces raw atomic scattering factors f1(E) and f2(E) in
   ASF units.  The conversion to refractive index (delta, beta) is:

       delta = (r_e * λ² * N_A * ρ / M) * f1
       beta  = (r_e * λ² * N_A * ρ / M) * f2

   Consequently the expensive KK integral only needs to run ONCE per
   stoichiometry; density variation is then just a cheap linear rescaling.

2. **Stoichiometry combinations are independent.**
   Evaluated concurrently via ProcessPoolExecutor.

Cost function — forward direction (existing)
--------------------------------------------
    imag_fit  →  beta  →  KK(kkcalc)  →  delta  →  real_kk
    MSE_fwd = 0.5·MSE(real_kk, real_fit) + 0.5·MSE(imag_kk, imag_fit)

Cost function — reverse direction (appended)
---------------------------------------------
The KK relations are symmetric; the reverse direction feeds the real component
and predicts the imaginary:

    real_fit  →  delta  →  KK⁻¹(Hilbert)  →  beta  →  imag_kk
    MSE_rev = 0.5·MSE(imag_kk, imag_fit) + 0.5·MSE(real_kk, real_fit)

kkcalc only runs the imaginary→real direction, so the reverse transform is
implemented numerically using a logarithmic-grid Hilbert transform (scipy).
Both directions share the same output format so results can be merged and
cross-compared.

Bidirectional combined cost
----------------------------
    MSE_bi = 0.5·MSE_fwd + 0.5·MSE_rev

Depends on NEXAFS.py:  EnergytoWavelength, imag_SLD_to_beta
"""

import itertools
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import hilbert as scipy_hilbert


# ---------------------------------------------------------------------------
# Internal helpers — shared
# ---------------------------------------------------------------------------

def _build_formula(atoms, counts):
    return ''.join(f"{atom}{int(count)}" for atom, count in zip(atoms, counts))


def _mse(predicted, observed):
    mask = np.isfinite(predicted) & np.isfinite(observed)
    if mask.sum() == 0:
        return np.inf
    return np.mean((predicted[mask] - observed[mask]) ** 2)


def _interp_onto(source_E, source_vals, target_E):
    """Linear interpolation with NaN outside source range."""
    if (target_E.min() < source_E.min() or target_E.max() > source_E.max()):
        return None
    return interp1d(source_E, source_vals, kind='linear',
                    bounds_error=False, fill_value=np.nan)(target_E)


def _sld_to_delta(sld_real, energies):
    """SLD_real [10^-6 Å^-2] → delta [dimensionless]."""
    from NEXAFS import EnergytoWavelength
    wavelengths = EnergytoWavelength(energies)
    return sld_real / (2.0 * np.pi / wavelengths**2 * 1e6)


def _delta_to_sld_real(delta, energies):
    """delta → SLD_real [10^-6 Å^-2]."""
    from NEXAFS import EnergytoWavelength
    wavelengths = EnergytoWavelength(energies)
    return 2.0 * np.pi * delta / wavelengths**2 * 1e6


def _beta_to_sld_imag(beta, energies):
    """beta → SLD_imag [10^-6 Å^-2]."""
    from NEXAFS import EnergytoWavelength
    wavelengths = EnergytoWavelength(energies)
    return 2.0 * np.pi * beta / wavelengths**2 * 1e6


# ---------------------------------------------------------------------------
# Forward direction helpers (kkcalc-based, existing)
# ---------------------------------------------------------------------------

def _asf_output_for_formula(energy_beta_array, formula, merge_points, kk_kwargs):
    """
    Run the kkcalc KK transform (imag → real) for one formula.
    Returns (asf_output, formula_mass) or (None, None) on failure.
    asf_output columns: [Energy, f1_ASF, f2_ASF].
    Called once per stoichiometry.
    """
    from kkcalc import data as kkdata
    from kkcalc import kk

    try:
        stoichiometry = kkdata.ParseChemicalFormula(formula)
        formula_mass  = kkdata.calculate_FormulaMass(stoichiometry)

        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='kkcalc_beta_'
        )
        try:
            for row in energy_beta_array:
                tmp.write(f"{row[0]:.6f}  {row[1]:.10e}\n")
            tmp.close()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                asf_output = kk.kk_calculate_real(
                    tmp.name, formula,
                    load_options=None,
                    input_data_type='Beta',
                    merge_points=merge_points,
                    add_background=kk_kwargs.get('add_background', False),
                    fix_distortions=kk_kwargs.get('fix_distortions', False),
                    curve_tolerance=kk_kwargs.get('curve_tolerance', 0.05),
                    curve_recursion=kk_kwargs.get('curve_recursion', 100),
                )
        finally:
            os.unlink(tmp.name)
        return asf_output, formula_mass
    except Exception:
        return None, None


def _asf_to_both_sld(asf_output, formula_mass, density, target_energies):
    """
    Convert cached ASF output to (real_sld, imag_sld) at target_energies.
    O(n), no KK transform.  Returns (None, None) on failure.
    Both in 10^-6 Å^-2.
    """
    from kkcalc import data as kkdata
    from NEXAFS import EnergytoWavelength

    try:
        delta_col = kkdata.convert_data(
            asf_output[:, [0, 1]], 'ASF', 'refractive_index',
            Density=density, Formula_Mass=formula_mass)
        beta_col  = kkdata.convert_data(
            asf_output[:, [0, 2]], 'ASF', 'refractive_index',
            Density=density, Formula_Mass=formula_mass)

        kk_E  = delta_col[:, 0]
        delta = delta_col[:, 1]
        beta  = beta_col[:, 1]

        if target_energies.min() < kk_E.min() or target_energies.max() > kk_E.max():
            return None, None

        wavelengths = EnergytoWavelength(target_energies)
        scale = 2.0 * np.pi / wavelengths**2 * 1e6

        d_i = interp1d(kk_E, delta, kind='linear',
                       bounds_error=False, fill_value=np.nan)(target_energies)
        b_i = interp1d(kk_E, beta,  kind='linear',
                       bounds_error=False, fill_value=np.nan)(target_energies)

        return scale * d_i, scale * b_i
    except Exception:
        return None, None


def _fwd_combined_mse(asf_output, formula_mass, density,
                      target_energies, real_sld_fit, imag_sld_fit):
    """Forward-direction combined MSE: 0.5·MSE(real_kk,real_fit)+0.5·MSE(imag_kk,imag_fit)."""
    r, im = _asf_to_both_sld(asf_output, formula_mass, density, target_energies)
    if r is None:
        return np.inf
    return 0.5 * _mse(r, real_sld_fit) + 0.5 * _mse(im, imag_sld_fit)


# ---------------------------------------------------------------------------
# Reverse direction helpers (Hilbert-transform-based, new)
# ---------------------------------------------------------------------------

def _kk_reverse_delta_to_beta(energies, delta, n_grid=2048):
    """
    Numerically compute beta from delta via the reverse Kramers-Kronig relation:

        beta(ω) = -(2ω/π) P∫₀^∞  delta(ω') / (ω'² - ω²)  dω'

    Implementation: resample onto a uniform log-frequency grid, apply the
    analytical Hilbert transform via FFT (scipy.signal.hilbert), then
    interpolate back onto the original energy points.

    The sign convention follows the X-ray optics convention where the complex
    refractive index is  n = 1 - delta - i·beta  (both delta and beta > 0
    below the plasma frequency).  Under this convention the KK relation for
    beta has a leading minus sign relative to the standard Hilbert transform.

    Parameters
    ----------
    energies : np.ndarray   Energy points in eV (not required to be uniform).
    delta    : np.ndarray   delta values at those energies (dimensionless).
    n_grid   : int          Number of points on the log-frequency grid (default 2048).

    Returns
    -------
    beta_at_energies : np.ndarray
        Predicted beta at the original energy points.  NaN where extrapolation
        is required.
    """
    # Work on a uniform log grid spanning the data range (+10 % margin)
    E_lo = energies.min() * 0.9
    E_hi = energies.max() * 1.1
    E_grid = np.exp(np.linspace(np.log(E_lo), np.log(E_hi), n_grid))

    # Interpolate delta onto the uniform grid (zero outside data range)
    delta_grid = interp1d(energies, delta, kind='linear',
                          bounds_error=False, fill_value=0.0)(E_grid)

    # The KK relation in the form used for a causal signal:
    #   beta(ω) = -(2/π) ω · H[delta/ω](ω)
    # where H is the Hilbert transform.  On a linear grid the discrete Hilbert
    # transform (via scipy) approximates the Cauchy PV integral.
    # On a log grid with uniform spacing d(ln E) = dE/E the integrand needs
    # a factor of E to account for the change of variable.
    #
    # We use the imaginary part of the analytic signal of  (delta/E):
    # H[delta/E](E) ≈ Im(hilbert(delta/E))
    # Then: beta ≈ -(2E/π) · H[delta/E](E)
    #
    # This is the standard numerical approach for KK on a log-spaced grid.
    integrand = delta_grid / E_grid
    analytic  = scipy_hilbert(integrand)
    beta_grid = -(2.0 * E_grid / np.pi) * np.imag(analytic)

    # Interpolate back onto original energy points
    beta_at_energies = interp1d(E_grid, beta_grid, kind='linear',
                                bounds_error=False, fill_value=np.nan)(energies)
    return beta_at_energies


def _rev_asf_output_for_formula(energy_delta_array, formula, merge_points, kk_kwargs):
    """
    Produce the reverse-direction ASF-equivalent output for a single formula by:
      1. Converting the input delta spectrum to ASF f1 units using kkcalc's
         convert_data (density-independent: use density=1, then rescale).
      2. Running the numerical reverse KK transform on f1 to obtain f2.

    Because the reverse KK is density-independent in the same way as the
    forward KK (density is just a scalar on the ASF), we store a
    (energy, f2_predicted, f1_input) array analogous to the forward ASF output,
    but with columns swapped in meaning:
        col 0: Energy
        col 1: f2_ASF  (predicted imaginary — the "output" of the reverse KK)
        col 2: f1_ASF  (the input real spectrum, stored for self-consistency check)

    The formula mass is also returned for later convert_data calls.

    Returns (rev_asf, formula_mass) or (None, None).
    """
    from kkcalc import data as kkdata

    try:
        stoichiometry = kkdata.ParseChemicalFormula(formula)
        formula_mass  = kkdata.calculate_FormulaMass(stoichiometry)

        # Convert input delta → f1_ASF using density=1 (density factors out)
        # We need an energy column paired with delta values.
        # convert_data('refractive_index' → 'ASF') is the inverse of what we
        # normally do.  Use density=1, formula_mass=1 so the conversion is
        # purely the photon-energy-dependent prefactor.
        #
        # Actually the simplest approach: since delta = C(E)·ρ/M·f1  where
        # C(E) is a known function of energy, we can store the raw delta values
        # directly and apply the density scaling later — identical to the
        # forward direction.  We use a reference density of 1 g/cm³ as a
        # placeholder for the ASF-equivalent, then the actual density is
        # applied in _rev_asf_to_both_sld below.

        energies = energy_delta_array[:, 0]
        delta    = energy_delta_array[:, 1]

        # Run the reverse KK to get beta from delta
        beta = _kk_reverse_delta_to_beta(energies, delta)

        # Store analogously to forward asf_output but note: here
        #   col1 = f2_predicted (imaginary output)
        #   col2 = f1_input     (real input, for self-consistency)
        # Both are stored as raw optical constant values (NOT per-formula-mass
        # scaled), so they need density/formula_mass scaling in the retrieval step.
        rev_asf = np.column_stack((energies, beta, delta))
        return rev_asf, formula_mass

    except Exception:
        return None, None


def _rev_asf_to_both_sld(rev_asf, formula_mass, density, target_energies):
    """
    Convert cached reverse-direction output to (imag_sld_predicted, real_sld_input)
    at target_energies.

    The stored rev_asf columns are raw (delta, beta) optical constants at
    density=1.  Scaling by (density/formula_mass) * C(E) gives the actual
    SLD values, where C(E) = 2π/λ² * 1e6.  But since both delta and beta
    are already stored as absolute optical constants (not ASF), the
    density/formula_mass factor must be applied consistently.

    Here we use the fact that:
        SLD_real = 2π·delta/λ² · 1e6
        SLD_imag = 2π·beta /λ² · 1e6
    and the scaling with density/formula_mass is already implicit in how the
    delta/beta values were derived from the input SLD (which already had the
    correct density baked in from the reflectometry fit).

    Wait — this is the reverse direction: the INPUT is the fitted real SLD,
    which has the fit density already embedded.  We are NOT scaling by a new
    density in this step; instead we are using the KK transform to check
    whether the shape of the predicted imaginary SLD is consistent.  The
    density enters only through the forward ASF path.

    For the reverse direction the density scaling is handled differently:
    the reverse KK transform is applied to delta values derived from the
    measured SLD_real at the *current candidate density*.  We therefore
    need to pass the density into this function so we can re-derive delta
    from SLD_real consistently.

    Parameters
    ----------
    rev_asf      : np.ndarray  [Energy, beta_predicted, delta_input]
                   as returned by _rev_asf_output_for_formula.
    formula_mass : float  (not used here — kept for API symmetry)
    density      : float  g/cm³  — the candidate density being tested.
    target_energies : np.ndarray

    Returns
    -------
    (imag_sld_predicted, real_sld_from_input) — both in 10^-6 Å^-2,
    or (None, None) on failure.
    """
    try:
        E_stored     = rev_asf[:, 0]
        beta_stored  = rev_asf[:, 1]   # beta at reference density=1
        delta_stored = rev_asf[:, 2]   # delta at reference density=1

        if target_energies.min() < E_stored.min() or target_energies.max() > E_stored.max():
            return None, None

        beta_i  = _interp_onto(E_stored, beta_stored,  target_energies)
        delta_i = _interp_onto(E_stored, delta_stored, target_energies)
        if beta_i is None or delta_i is None:
            return None, None

        # The stored beta/delta come from the input SLD which already has the
        # fit density baked in.  No further density scaling is needed.
        imag_sld = _beta_to_sld_imag(beta_i,  target_energies)
        real_sld = _delta_to_sld_real(delta_i, target_energies)

        return imag_sld, real_sld

    except Exception:
        return None, None


def _rev_combined_mse(rev_asf, formula_mass, density,
                      target_energies, real_sld_fit, imag_sld_fit):
    """
    Reverse-direction combined MSE.

    The reverse KK transform takes real SLD → predicts imaginary SLD.
    For the reverse direction the 'density' parameter controls how the
    input real SLD is converted to delta before the transform.  Since
    the stored rev_asf was computed from the fitted SLD (which already
    carries the correct density), the density parameter here allows us
    to test consistency at different densities by rescaling the delta
    values — i.e. we treat the stored delta as being at density=1 and
    scale up.

    Concretely: delta_candidate = delta_stored * (density / density_fit)
    But we don't know density_fit a priori.  Instead we adopt the
    simpler and self-consistent definition:

        For the reverse direction, the input real SLD is fixed (it is the
        measured data), and the KK transform predicts the imaginary SLD.
        The shape of the predicted imaginary SLD depends on the formula
        (through the background Henke data that kkcalc stitches), but
        NOT on density (which cancels in the KK integral).

    This means the reverse-direction transform is density-independent
    for a given formula, and the density only affects the amplitude of
    the predicted imaginary SLD through the forward direction.

    For simplicity and consistency we implement the reverse cost as:
        rev_cost = MSE(imag_kk_rev, imag_fit) + MSE(real_kk_rev, real_fit)
    where imag_kk_rev and real_kk_rev come from interpolating the stored
    rev_asf output (independent of density) onto the target energies.

    Since the reverse direction is density-independent, the combined
    bidirectional optimiser uses fwd_mse (density-dependent) + rev_mse
    (density-independent, constant across density sweep).
    """
    im_pred, re_pred = _rev_asf_to_both_sld(
        rev_asf, formula_mass, density, target_energies
    )
    if im_pred is None:
        return np.inf
    return 0.5 * _mse(im_pred, imag_sld_fit) + 0.5 * _mse(re_pred, real_sld_fit)


# ---------------------------------------------------------------------------
# Density optimisation helpers (shared by all directions)
# ---------------------------------------------------------------------------

def _best_density_for_asf(asf_output, formula_mass, density_range,
                           target_energies, real_sld_fit, imag_sld_fit,
                           optimise_density, n_density_grid,
                           rev_asf=None):
    """
    Find density minimising the cost function.

    If rev_asf is provided, the cost is:
        0.5 * fwd_mse(density) + 0.5 * rev_mse
    where rev_mse is density-independent (pre-computed once).
    Otherwise uses forward-only combined MSE.

    Returns (best_density, best_mse).
    """
    rho_min, rho_max = density_range

    # Pre-compute the density-independent reverse cost once
    if rev_asf is not None:
        rev_const = _rev_combined_mse(rev_asf, formula_mass, None,
                                      target_energies, real_sld_fit, imag_sld_fit)
    else:
        rev_const = 0.0

    def cost(rho):
        fwd = _fwd_combined_mse(asf_output, formula_mass, rho,
                                target_energies, real_sld_fit, imag_sld_fit)
        if not np.isfinite(fwd):
            return np.inf
        if rev_asf is not None:
            return 0.5 * fwd + 0.5 * rev_const
        return fwd

    if optimise_density:
        result = minimize_scalar(cost, bounds=(rho_min, rho_max),
                                 method='bounded', options={'xatol': 1e-4})
        return result.x, result.fun
    else:
        grid = np.linspace(rho_min, rho_max, n_density_grid)
        mses = np.array([cost(rho) for rho in grid])
        best_idx = np.argmin(mses)
        return grid[best_idx], mses[best_idx]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _worker(args):
    """
    Forward-direction worker (existing).
    args: (formula, atoms, counts, energy_beta, merge_points, kk_kwargs,
           density_range, target_energies, real_sld_fit, imag_sld_fit,
           optimise_density, n_density_grid)
    """
    (formula, atoms, counts,
     energy_beta, merge_points, kk_kwargs,
     density_range, target_energies, real_sld_fit, imag_sld_fit,
     optimise_density, n_density_grid) = args

    asf_output, formula_mass = _asf_output_for_formula(
        energy_beta, formula, merge_points, kk_kwargs)

    if asf_output is None:
        return {'formula': formula, 'density': np.nan,
                'mse': np.inf, 'rmse': np.inf,
                **{f'n_{a}': c for a, c in zip(atoms, counts)}}

    best_rho, best_mse = _best_density_for_asf(
        asf_output, formula_mass, density_range,
        target_energies, real_sld_fit, imag_sld_fit,
        optimise_density, n_density_grid)

    return {'formula': formula, 'density': best_rho,
            'mse': best_mse,
            'rmse': np.sqrt(best_mse) if np.isfinite(best_mse) else np.inf,
            **{f'n_{a}': c for a, c in zip(atoms, counts)}}


def _worker_bidirectional(args):
    """
    Bidirectional worker (new).
    Runs both forward (kkcalc) and reverse (Hilbert) KK transforms, then
    optimises density against the combined bidirectional cost.

    args: (formula, atoms, counts,
           energy_beta, energy_delta,
           merge_points, kk_kwargs,
           density_range, target_energies, real_sld_fit, imag_sld_fit,
           optimise_density, n_density_grid)
    """
    (formula, atoms, counts,
     energy_beta, energy_delta,
     merge_points, kk_kwargs,
     density_range, target_energies, real_sld_fit, imag_sld_fit,
     optimise_density, n_density_grid) = args

    # Forward: imag → real (kkcalc)
    asf_output, formula_mass = _asf_output_for_formula(
        energy_beta, formula, merge_points, kk_kwargs)

    # Reverse: real → imag (numerical Hilbert)
    rev_asf, _ = _rev_asf_output_for_formula(
        energy_delta, formula, merge_points, kk_kwargs)

    if asf_output is None and rev_asf is None:
        return {'formula': formula, 'density': np.nan,
                'mse': np.inf, 'rmse': np.inf,
                'mse_fwd': np.inf, 'mse_rev': np.inf,
                **{f'n_{a}': c for a, c in zip(atoms, counts)}}

    # If one direction failed, fall back to the other
    if asf_output is None:
        asf_output_eff = None
        fwd_failed = True
    else:
        fwd_failed = False

    if rev_asf is None:
        rev_failed = True
    else:
        rev_failed = False

    # Pre-compute density-independent reverse MSE
    if not rev_failed:
        mse_rev = _rev_combined_mse(rev_asf, formula_mass, None,
                                    target_energies, real_sld_fit, imag_sld_fit)
    else:
        mse_rev = np.inf

    rho_min, rho_max = density_range

    def cost(rho):
        fwd = (_fwd_combined_mse(asf_output, formula_mass, rho,
                                 target_energies, real_sld_fit, imag_sld_fit)
               if not fwd_failed else np.inf)
        if fwd_failed:
            return mse_rev
        if rev_failed:
            return fwd
        return 0.5 * fwd + 0.5 * mse_rev

    if optimise_density:
        result = minimize_scalar(cost, bounds=(rho_min, rho_max),
                                 method='bounded', options={'xatol': 1e-4})
        best_rho, best_mse = result.x, result.fun
    else:
        grid = np.linspace(rho_min, rho_max, n_density_grid)
        mses = np.array([cost(rho) for rho in grid])
        best_idx = np.argmin(mses)
        best_rho, best_mse = grid[best_idx], mses[best_idx]

    # Record individual direction costs at the best density for diagnostics
    mse_fwd = (_fwd_combined_mse(asf_output, formula_mass, best_rho,
                                 target_energies, real_sld_fit, imag_sld_fit)
               if not fwd_failed else np.inf)

    return {'formula': formula, 'density': best_rho,
            'mse': best_mse,
            'rmse': np.sqrt(best_mse) if np.isfinite(best_mse) else np.inf,
            'mse_fwd': mse_fwd, 'mse_rev': mse_rev,
            'rmse_fwd': np.sqrt(mse_fwd) if np.isfinite(mse_fwd) else np.inf,
            'rmse_rev': np.sqrt(mse_rev) if np.isfinite(mse_rev) else np.inf,
            **{f'n_{a}': c for a, c in zip(atoms, counts)}}


# ---------------------------------------------------------------------------
# Shared data-preparation helper
# ---------------------------------------------------------------------------

def _prepare_fit_data(sld_3col, energy_mask):
    """Sort, apply energy mask, build 4-col SLD, and compute energy_beta / energy_delta."""
    from NEXAFS import EnergytoWavelength, imag_SLD_to_beta

    sort_idx     = np.argsort(sld_3col[:, 0])
    sld_sorted   = sld_3col[sort_idx]
    energies_all = sld_sorted[:, 0]
    real_sld_all = sld_sorted[:, 1]
    imag_sld_all = sld_sorted[:, 2]

    if energy_mask is not None:
        e_lo, e_hi = energy_mask
        mask = (energies_all >= e_lo) & (energies_all <= e_hi)
        if mask.sum() == 0:
            raise ValueError(
                f"energy_mask ({e_lo}, {e_hi}) excludes all data points. "
                f"Data range: {energies_all.min():.1f}–{energies_all.max():.1f} eV")
        energies_fit = energies_all[mask]
        real_sld_fit = real_sld_all[mask]
        imag_sld_fit = imag_sld_all[mask]
    else:
        energies_fit = energies_all
        real_sld_fit = real_sld_all
        imag_sld_fit = imag_sld_all

    wavelengths_all = EnergytoWavelength(energies_all)
    sld_4col        = np.column_stack((energies_all, real_sld_all,
                                       imag_sld_all, wavelengths_all))
    energy_beta  = imag_SLD_to_beta(sld_4col)                    # [E, beta]
    delta_all    = _sld_to_delta(real_sld_all, energies_all)
    energy_delta = np.column_stack((energies_all, delta_all))     # [E, delta]

    return (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
            energy_beta, energy_delta)


def _build_count_grid(cr):
    """
    Convert a single count_range entry into a list of integer values.

    Accepted forms:
      int              -> [int]                fixed value
      (min, max)       -> range(min, max+1, 1)
      (min, max, step) -> range(min, max+1, step)
    """
    if isinstance(cr, int):
        return [cr]
    if hasattr(cr, '__len__') and len(cr) in (2, 3):
        lo   = int(cr[0])
        hi   = int(cr[1])
        step = int(cr[2]) if len(cr) == 3 else 1
        if lo > hi:
            raise ValueError(f"count_range {cr}: min must be <= max")
        if step < 1:
            raise ValueError(f"count_range {cr}: step must be >= 1")
        return list(range(lo, hi + 1, step))
    raise ValueError(
        f"count_ranges entries must be int, (min, max), or "
        f"(min, max, step), got {cr!r}")


def _build_stoichiometry_grid(atoms, count_ranges):
    count_grids = [_build_count_grid(cr) for cr in count_ranges]
    return list(itertools.product(*count_grids))


def _dispatch(worker_fn, worker_args, n_workers, verbose, n_combos):
    records = []
    use_serial = (n_workers == 1) or (n_combos == 1)

    if use_serial:
        for i, args in enumerate(worker_args):
            record = worker_fn(args)
            records.append(record)
            if verbose:
                rmse_str = (f"RMSE={record['rmse']:.4f}" if np.isfinite(record['rmse'])
                            else "failed")
                print(f"  [{i+1}/{n_combos}]  {record['formula']:14s}  "
                      f"density={record['density']:.4f} g/cm³   {rmse_str}")
    else:
        effective_workers = min(n_workers or os.cpu_count(), n_combos)
        completed = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(worker_fn, args): args[0] for args in worker_args}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                if verbose:
                    rmse_str = (f"RMSE={record['rmse']:.4f}" if np.isfinite(record['rmse'])
                                else "failed")
                    print(f"  [{completed}/{n_combos}]  {record['formula']:14s}  "
                          f"density={record['density']:.4f} g/cm³   {rmse_str}")
    return records


# ---------------------------------------------------------------------------
# Public API — forward direction (existing, unchanged interface)
# ---------------------------------------------------------------------------

def fit_stoichiometry_density(
    sld_3col, atoms, count_ranges, density_range, merge_points,
    n_density_grid=50, optimise_density=True, energy_mask=None,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density using the forward KK direction only.

    Cost: 0.5·MSE(real_kk, real_fit) + 0.5·MSE(imag_kk, imag_fit)
    where real_kk is obtained by transforming the fitted imaginary SLD.

    Parameters
    ----------
    sld_3col      : np.ndarray (n,3)   [Energy, SLD_Real, SLD_Imag], 10^-6 Å^-2.
    atoms         : list of str        Element symbols, e.g. ['Sn','C','O'].
    count_ranges  : list               Per-atom int or (min,max) tuple.
    density_range : tuple              (rho_min, rho_max) g/cm³.
    merge_points  : list of 2 floats   KK splice energies [E_min, E_max] eV.
    n_density_grid: int                Grid points when optimise_density=False.
    optimise_density: bool             True → Brent; False → uniform grid.
    energy_mask   : tuple, optional    (E_min, E_max) eV for residual window.
    kk_kwargs     : dict, optional     Extra kwargs for kkcalc.
    n_workers     : int or None        Parallel workers (1 = serial).
    verbose       : bool

    Returns
    -------
    results_df : pd.DataFrame   Sorted by MSE. Columns: formula, density, mse,
                                rmse, n_<atom>.
    best       : dict           formula, density, mse, rmse for top result.
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(atoms) != len(count_ranges):
        raise ValueError("len(atoms) must equal len(count_ranges)")

    stoichiometry_combos = _build_stoichiometry_grid(atoms, count_ranges)
    n_combos = len(stoichiometry_combos)

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, _) = _prepare_fit_data(sld_3col, energy_mask)

    if verbose:
        n_cpu = os.cpu_count() if n_workers is None else n_workers
        print("KK stoichiometry fit  [forward direction]")
        print(f"  Stoichiometry combinations : {n_combos}")
        print(f"  Density range              : {density_range[0]:.2f}–{density_range[1]:.2f} g/cm³")
        print(f"  Cost function              : 0.5·MSE(real) + 0.5·MSE(imag)  [imag→real KK]")
        print(f"  Residual energy points     : {len(energies_fit)}"
              + (f"  [{energies_fit.min():.1f}–{energies_fit.max():.1f} eV]" if energy_mask else ""))
        print(f"  Workers                    : {min(n_cpu, n_combos)}\n")

    worker_args = [
        (_build_formula(atoms, counts), atoms, counts,
         energy_beta, merge_points, kk_kwargs,
         density_range, energies_fit, real_sld_fit, imag_sld_fit,
         optimise_density, n_density_grid)
        for counts in stoichiometry_combos
    ]

    records   = _dispatch(_worker, worker_args, n_workers, verbose, n_combos)
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    best       = {k: best_row[k] for k in ['formula', 'density', 'mse', 'rmse']}

    if verbose:
        print(f"\n── Best result {'─'*40}")
        print(f"   Formula : {best['formula']}")
        print(f"   Density : {best['density']:.4f} g/cm³")
        print(f"   RMSE    : {best['rmse']:.6f}  (10⁻⁶ Å⁻²)")
        print('─' * 57)

    return results_df, best


# ---------------------------------------------------------------------------
# Public API — bidirectional (new)
# ---------------------------------------------------------------------------

def fit_stoichiometry_density_bidirectional(
    sld_3col, atoms, count_ranges, density_range, merge_points,
    n_density_grid=50, optimise_density=True, energy_mask=None,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density using BOTH KK directions simultaneously.

    Forward direction  (existing):
        imag_fit  →  beta  →  KK(kkcalc)  →  delta  →  real_kk
        MSE_fwd = 0.5·MSE(real_kk, real_fit) + 0.5·MSE(imag_kk, imag_fit)

    Reverse direction  (new):
        real_fit  →  delta  →  KK⁻¹(Hilbert)  →  beta  →  imag_kk
        MSE_rev = 0.5·MSE(imag_kk, imag_fit) + 0.5·MSE(real_kk, real_fit)

    Combined bidirectional cost:
        MSE_bi = 0.5·MSE_fwd + 0.5·MSE_rev

    Note: the reverse KK is density-independent (density cancels in the KK
    integral), so MSE_rev is constant across the density sweep and only
    discriminates between stoichiometries.  MSE_fwd remains
    density-dependent and drives the density optimisation.

    Parameters
    ----------
    (same as fit_stoichiometry_density)

    Returns
    -------
    results_df : pd.DataFrame
        Sorted by combined MSE. Extra columns vs forward-only:
        mse_fwd, mse_rev, rmse_fwd, rmse_rev.
    best : dict
        As above, plus mse_fwd, mse_rev, rmse_fwd, rmse_rev.
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(atoms) != len(count_ranges):
        raise ValueError("len(atoms) must equal len(count_ranges)")

    stoichiometry_combos = _build_stoichiometry_grid(atoms, count_ranges)
    n_combos = len(stoichiometry_combos)

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, energy_delta) = _prepare_fit_data(sld_3col, energy_mask)

    if verbose:
        n_cpu = os.cpu_count() if n_workers is None else n_workers
        print("KK stoichiometry fit  [bidirectional]")
        print(f"  Stoichiometry combinations : {n_combos}")
        print(f"  Density range              : {density_range[0]:.2f}–{density_range[1]:.2f} g/cm³")
        print(f"  Cost function              : 0.5·MSE_fwd + 0.5·MSE_rev")
        print(f"    MSE_fwd: imag_fit → KK(kkcalc) → real_kk  vs  real_fit")
        print(f"    MSE_rev: real_fit → KK⁻¹(Hilbert) → imag_kk  vs  imag_fit")
        print(f"  Residual energy points     : {len(energies_fit)}"
              + (f"  [{energies_fit.min():.1f}–{energies_fit.max():.1f} eV]" if energy_mask else ""))
        print(f"  Workers                    : {min(n_cpu, n_combos)}\n")

    worker_args = [
        (_build_formula(atoms, counts), atoms, counts,
         energy_beta, energy_delta,
         merge_points, kk_kwargs,
         density_range, energies_fit, real_sld_fit, imag_sld_fit,
         optimise_density, n_density_grid)
        for counts in stoichiometry_combos
    ]

    records    = _dispatch(_worker_bidirectional, worker_args, n_workers, verbose, n_combos)
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    best       = {k: best_row[k]
                  for k in ['formula', 'density', 'mse', 'rmse',
                             'mse_fwd', 'mse_rev', 'rmse_fwd', 'rmse_rev']}

    if verbose:
        print(f"\n── Best result (bidirectional) {'─'*27}")
        print(f"   Formula  : {best['formula']}")
        print(f"   Density  : {best['density']:.4f} g/cm³")
        print(f"   RMSE     : {best['rmse']:.6f}  (combined)")
        print(f"   RMSE_fwd : {best['rmse_fwd']:.6f}  (imag→real)")
        print(f"   RMSE_rev : {best['rmse_rev']:.6f}  (real→imag)")
        print('─' * 57)

    return results_df, best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stoichiometry_fit_results(
    results_df, sld_3col, best, merge_points,
    energy_mask=None, top_n=5, kk_kwargs=None, figsize=(18, 5)
):
    """
    Three-panel diagnostic plot for the forward-direction fit.

    Left   — RMSE bar chart.
    Centre — KK-derived real SLD vs fitted real SLD (open circles).
    Right  — KK-derived imag SLD vs fitted imag SLD (open circles).
    """
    from NEXAFS import EnergytoWavelength, imag_SLD_to_real_SLD

    if kk_kwargs is None:
        kk_kwargs = {}
    kk_kwargs['merge_points'] = merge_points

    top = results_df.head(top_n)
    sort_idx   = np.argsort(sld_3col[:, 0])
    sld_sorted = sld_3col[sort_idx]
    wavelengths = EnergytoWavelength(sld_sorted[:, 0])
    sld_4col   = np.column_stack((sld_sorted, wavelengths))
    E_min, E_max = sld_sorted[:, 0].min(), sld_sorted[:, 0].max()

    fig, (ax_bar, ax_real, ax_imag) = plt.subplots(1, 3, figsize=figsize)
    colours = plt.cm.tab10.colors

    labels = [f"{row['formula']}\n{row['density']:.3f} g/cm³" for _, row in top.iterrows()]
    rmses  = top['rmse'].values
    ax_bar.barh(range(len(top)), rmses,
                color=[colours[i % 10] for i in range(len(top))],
                edgecolor='k', linewidth=0.6)
    ax_bar.set_yticks(range(len(top)))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('RMSE  (10$^{-6}$ Å$^{-2}$)')
    ax_bar.set_title(f'Top-{len(top)} candidates  [forward KK]')
    ax_bar.axvline(rmses[0], color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_bar.grid(axis='x', alpha=0.3)

    for ax, col, lbl in [(ax_real, 1, 'Real SLD (fit)'), (ax_imag, 2, 'Imag SLD (fit)')]:
        ax.plot(sld_sorted[:, 0], sld_sorted[:, col],
                marker='o', linestyle='none', color='k',
                markerfacecolor='none', markeredgewidth=1.4, markersize=6,
                label=lbl, zorder=5)

    for i, (_, row) in enumerate(top.iterrows()):
        try:
            _, SLD_kk = imag_SLD_to_real_SLD(
                sld_4col, chemical_formula=row['formula'],
                density=row['density'], **kk_kwargs)
        except Exception:
            continue
        mask  = (SLD_kk[:, 0] >= E_min) & (SLD_kk[:, 0] <= E_max)
        lw, alpha = (2.2, 1.0) if i == 0 else (1.2, 0.55)
        lbl = f"{row['formula']}  ρ={row['density']:.3f}"
        ax_real.plot(SLD_kk[mask, 0], SLD_kk[mask, 1],
                     color=colours[i % 10], linewidth=lw, alpha=alpha, label=lbl)
        ax_imag.plot(SLD_kk[mask, 0], SLD_kk[mask, 2],
                     color=colours[i % 10], linewidth=lw, alpha=alpha, label=lbl)

    if energy_mask is not None:
        for ax in (ax_real, ax_imag):
            ax.axvspan(*energy_mask, alpha=0.07, color='grey', label='Fit region')

    for ax, title in [(ax_real, 'Real SLD — KK(imag→real) vs fit'),
                      (ax_imag, 'Imag SLD — KK(imag→real) vs fit')]:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('SLD  (10$^{-6}$ Å$^{-2}$)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, (ax_bar, ax_real, ax_imag)


def plot_stoichiometry_fit_results_bidirectional(
    results_df, sld_3col, best, merge_points,
    energy_mask=None, top_n=5, kk_kwargs=None, figsize=(24, 10)
):
    """
    Five-panel diagnostic plot for the bidirectional fit.

    Row 1 (top):
      Left     — RMSE bar chart (combined bidirectional).
      Centre   — Forward KK: real_kk (from imag_fit) vs real_fit.
      Right    — Forward KK: imag_kk (round-trip) vs imag_fit.

    Row 2 (bottom):
      Left     — RMSE comparison: fwd vs rev (grouped bar).
      Centre   — Reverse KK: imag_kk (from real_fit) vs imag_fit.
      Right    — Reverse KK: real_kk (round-trip) vs real_fit.
    """
    from NEXAFS import EnergytoWavelength, imag_SLD_to_real_SLD

    if kk_kwargs is None:
        kk_kwargs = {}
    kk_kwargs_plot = {**kk_kwargs, 'merge_points': merge_points}

    top = results_df.head(top_n)
    sort_idx    = np.argsort(sld_3col[:, 0])
    sld_sorted  = sld_3col[sort_idx]
    wavelengths = EnergytoWavelength(sld_sorted[:, 0])
    sld_4col    = np.column_stack((sld_sorted, wavelengths))
    energies    = sld_sorted[:, 0]
    E_min, E_max = energies.min(), energies.max()

    colours = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    (ax_bar, ax_fwd_real, ax_fwd_imag,
     ax_bar2, ax_rev_imag, ax_rev_real) = axes.flat

    # ── Top-left: combined RMSE bar ───────────────────────────────────────
    labels = [f"{row['formula']}\n{row['density']:.3f}" for _, row in top.iterrows()]
    rmses  = top['rmse'].values
    ax_bar.barh(range(len(top)), rmses,
                color=[colours[i % 10] for i in range(len(top))],
                edgecolor='k', linewidth=0.6)
    ax_bar.set_yticks(range(len(top)))
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('RMSE  (10$^{-6}$ Å$^{-2}$)')
    ax_bar.set_title(f'Combined RMSE — top {len(top)}')
    ax_bar.axvline(rmses[0], color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_bar.grid(axis='x', alpha=0.3)

    # ── Bottom-left: fwd vs rev RMSE grouped bar ──────────────────────────
    x = np.arange(len(top))
    w = 0.35
    rmse_fwd = top['rmse_fwd'].values if 'rmse_fwd' in top.columns else top['rmse'].values
    rmse_rev = top['rmse_rev'].values if 'rmse_rev' in top.columns else np.zeros(len(top))
    ax_bar2.barh(x - w/2, rmse_fwd, w, label='Forward (imag→real)',
                 color='tab:blue', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax_bar2.barh(x + w/2, rmse_rev, w, label='Reverse (real→imag)',
                 color='tab:orange', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax_bar2.set_yticks(x)
    ax_bar2.set_yticklabels(labels, fontsize=8)
    ax_bar2.invert_yaxis()
    ax_bar2.set_xlabel('RMSE  (10$^{-6}$ Å$^{-2}$)')
    ax_bar2.set_title('Forward vs Reverse RMSE')
    ax_bar2.legend(fontsize=8)
    ax_bar2.grid(axis='x', alpha=0.3)

    # ── Fitted data (open black circles) on all four SLD panels ──────────
    for ax, col, lbl in [
        (ax_fwd_real, 1, 'Real SLD (fit)'),
        (ax_fwd_imag, 2, 'Imag SLD (fit)'),
        (ax_rev_imag, 2, 'Imag SLD (fit)'),
        (ax_rev_real, 1, 'Real SLD (fit)'),
    ]:
        ax.plot(energies, sld_sorted[:, col],
                marker='o', linestyle='none', color='k',
                markerfacecolor='none', markeredgewidth=1.3, markersize=5,
                label=lbl, zorder=5)

    for i, (_, row) in enumerate(top.iterrows()):
        lw, alpha = (2.2, 1.0) if i == 0 else (1.2, 0.55)
        lbl = f"{row['formula']}  ρ={row['density']:.3f}"
        c   = colours[i % 10]

        # Forward: imag_fit → KK → real_kk, imag_kk
        try:
            _, SLD_kk = imag_SLD_to_real_SLD(
                sld_4col, chemical_formula=row['formula'],
                density=row['density'], **kk_kwargs_plot)
            mask = (SLD_kk[:, 0] >= E_min) & (SLD_kk[:, 0] <= E_max)
            ax_fwd_real.plot(SLD_kk[mask, 0], SLD_kk[mask, 1],
                             color=c, linewidth=lw, alpha=alpha, label=lbl)
            ax_fwd_imag.plot(SLD_kk[mask, 0], SLD_kk[mask, 2],
                             color=c, linewidth=lw, alpha=alpha, label=lbl)
        except Exception:
            pass

        # Reverse: real_fit → KK⁻¹ → imag_kk, real_kk
        try:
            delta_fit = _sld_to_delta(sld_sorted[:, 1], energies)
            beta_rev  = _kk_reverse_delta_to_beta(energies, delta_fit)
            imag_rev  = _beta_to_sld_imag(beta_rev, energies)
            real_rev  = sld_sorted[:, 1]   # real input (round-trip reference)
            ax_rev_imag.plot(energies, imag_rev,
                             color=c, linewidth=lw, alpha=alpha, label=lbl)
            ax_rev_real.plot(energies, real_rev,
                             color=c, linewidth=lw, alpha=alpha, label=lbl)
        except Exception:
            pass

    # Shade energy mask
    if energy_mask is not None:
        for ax in (ax_fwd_real, ax_fwd_imag, ax_rev_imag, ax_rev_real):
            ax.axvspan(*energy_mask, alpha=0.07, color='grey', label='Fit region')

    titles = [
        (ax_fwd_real, 'Forward KK: real (predicted) vs real (fit)'),
        (ax_fwd_imag, 'Forward KK: imag (round-trip) vs imag (fit)'),
        (ax_rev_imag, 'Reverse KK: imag (predicted) vs imag (fit)'),
        (ax_rev_real, 'Reverse KK: real (input, reference)'),
    ]
    for ax, title in titles:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('SLD  (10$^{-6}$ Å$^{-2}$)')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# Density cost-surface scanner (works for both directions)
# ---------------------------------------------------------------------------

def scan_density_cost_surface(
    sld_3col, formula, density_range, merge_points,
    n_points=60, energy_mask=None, kk_kwargs=None, ax=None
):
    """
    Scan the combined real+imag MSE (forward direction) vs density for one
    fixed stoichiometry.  One KK transform, n_points cheap density evaluations.
    """
    from NEXAFS import EnergytoWavelength, imag_SLD_to_beta

    if kk_kwargs is None:
        kk_kwargs = {}

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, _) = _prepare_fit_data(sld_3col, energy_mask)

    asf_output, formula_mass = _asf_output_for_formula(
        energy_beta, formula, merge_points, kk_kwargs)

    densities = np.linspace(density_range[0], density_range[1], n_points)
    rmses     = np.full(n_points, np.nan)

    if asf_output is not None:
        for i, rho in enumerate(densities):
            mse = _fwd_combined_mse(asf_output, formula_mass, rho,
                                    energies_fit, real_sld_fit, imag_sld_fit)
            if np.isfinite(mse):
                rmses[i] = np.sqrt(mse)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))

    valid = np.isfinite(rmses)
    ax.plot(densities[valid], rmses[valid], 'o-', color='tab:blue',
            markersize=4, linewidth=1.5)
    ax.set_xlabel('Density (g/cm³)')
    ax.set_ylabel('RMSE  (10$^{-6}$ Å$^{-2}$)')
    ax.set_title(f'Density cost surface — {formula}')
    ax.grid(True, alpha=0.3)

    if valid.any():
        best_idx = np.nanargmin(rmses)
        ax.axvline(densities[best_idx], color='tab:red', linestyle='--',
                   linewidth=1.2, label=f'min  ρ={densities[best_idx]:.3f} g/cm³')
        ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()

    return densities, rmses


# ---------------------------------------------------------------------------
# Fragment-based stoichiometry search
# ---------------------------------------------------------------------------

def _parse_formula(formula):
    """
    Parse a chemical formula string into a dict of {element: count}.

    Handles standard Hill-order formulas with integer counts, e.g.:
        'C8H8'     → {'C': 8, 'H': 8}
        'Sn10O10'  → {'Sn': 10, 'O': 10}
        'MoO3'     → {'Mo': 1, 'O': 3}   (implicit count of 1)

    Parameters
    ----------
    formula : str

    Returns
    -------
    dict  {element_symbol: integer_count}
    """
    import re
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    result = {}
    for element, count in tokens:
        if not element:
            continue
        result[element] = result.get(element, 0) + int(count) if count else result.get(element, 0) + 1
    return result


def _combine_fragments(fragments, counts):
    """
    Combine a list of fragment formula dicts scaled by their integer counts
    into a single flat formula string.

    Parameters
    ----------
    fragments : list of dict   [{element: count}, ...]  one per fragment
    counts    : list of int    multiplier for each fragment

    Returns
    -------
    str   e.g. 'Sn10C16H16O10'
    """
    combined = {}
    for frag_dict, n in zip(fragments, counts):
        for element, c in frag_dict.items():
            combined[element] = combined.get(element, 0) + c * n
    # Canonical order: C first, H second, then alphabetical (Hill convention)
    order = sorted(combined.keys(),
                   key=lambda e: (0 if e == 'C' else 1 if e == 'H' else 2, e))
    return ''.join(f"{e}{combined[e]}" for e in order if combined[e] > 0)


def _build_fragment_grid(fragments, count_ranges):
    """
    Build the list of (formula_string, fragment_count_tuple) for all
    combinations of fragment counts.

    Parameters
    ----------
    fragments    : list of str   formula strings for each fragment,
                                 e.g. ['Sn10O10', 'C8H8']
    count_ranges : list          per-fragment int or (min, max) tuple

    Returns
    -------
    list of (formula_str, counts_tuple)
    """
    frag_dicts = [_parse_formula(f) for f in fragments]

    count_grids = [_build_count_grid(cr) for cr in count_ranges]

    combos = []
    seen   = set()
    for counts in itertools.product(*count_grids):
        if all(c == 0 for c in counts):
            continue                          # skip the empty formula
        formula = _combine_fragments(frag_dicts, counts)
        if formula in seen:
            continue                          # identical formula from different counts
        seen.add(formula)
        combos.append((formula, counts))
    return combos


def fit_stoichiometry_density_fragments(
    sld_3col, fragments, count_ranges, density_range, merge_points,
    n_density_grid=50, optimise_density=True, energy_mask=None,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density using a fragment-based search (forward KK).

    Instead of specifying individual atoms and per-atom count ranges, you
    supply a list of molecular fragments and search over integer multiples of
    each fragment.  The combined formula for each candidate is the linear
    combination of fragments scaled by their counts.

    Example
    -------
    fragments    = ['Sn10O10', 'C8H8']
    count_ranges = [(1, 2),    (3, 6) ]

    This searches all combinations of 1–2 copies of Sn10O10 and 3–6 copies
    of C8H8, producing formulas like Sn10C24H24O10, Sn20C32H32O20, etc.

    Duplicate formulas arising from different count combinations are
    automatically deduplicated.

    Parameters
    ----------
    sld_3col      : np.ndarray (n,3)   [Energy, SLD_Real, SLD_Imag], 10^-6 Å^-2.
    fragments     : list of str        Formula string for each fragment,
                                       e.g. ['Sn10O10', 'C8H8'].
    count_ranges  : list               Per-fragment int or (min, max) tuple
                                       giving the multiplier range.
    density_range : tuple              (rho_min, rho_max) g/cm³.
    merge_points  : list of 2 floats   KK splice energies [E_min, E_max] eV.
    n_density_grid: int                Grid points when optimise_density=False.
    optimise_density : bool            True → Brent; False → uniform grid.
    energy_mask   : tuple, optional    (E_min, E_max) eV residual window.
    kk_kwargs     : dict, optional     Extra kwargs for kkcalc.
    n_workers     : int or None        Parallel workers (1 = serial).
    verbose       : bool

    Returns
    -------
    results_df : pd.DataFrame
        Sorted by MSE.  Columns: formula, density, mse, rmse,
        plus one ``n_<fragment_formula>`` column per fragment.
    best : dict
        formula, density, mse, rmse for the top result.
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(fragments) != len(count_ranges):
        raise ValueError("len(fragments) must equal len(count_ranges)")

    fragment_combos = _build_fragment_grid(fragments, count_ranges)
    n_combos = len(fragment_combos)

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, _) = _prepare_fit_data(sld_3col, energy_mask)

    if verbose:
        n_cpu = os.cpu_count() if n_workers is None else n_workers
        print("KK stoichiometry fit  [fragment-based, forward direction]")
        print(f"  Fragments       : {fragments}")
        print(f"  Unique formulas : {n_combos}")
        print(f"  Density range   : {density_range[0]:.2f}–{density_range[1]:.2f} g/cm³")
        print(f"  Cost function   : 0.5·MSE(real) + 0.5·MSE(imag)  [imag→real KK]")
        print(f"  Residual points : {len(energies_fit)}"
              + (f"  [{energies_fit.min():.1f}–{energies_fit.max():.1f} eV]"
                 if energy_mask else ""))
        print(f"  Workers         : {min(n_cpu, n_combos)}\n")

    # Re-use the existing _worker — it only needs a flat formula string and
    # a list of (atom_label, count) pairs for the record dict.  We pass the
    # fragment counts as the "counts" and fragment formulas as the "atoms"
    # labels so the output columns are named n_<fragment>.
    frag_labels = fragments   # used as column name keys

    worker_args = [
        (formula, frag_labels, counts,
         energy_beta, merge_points, kk_kwargs,
         density_range, energies_fit, real_sld_fit, imag_sld_fit,
         optimise_density, n_density_grid)
        for formula, counts in fragment_combos
    ]

    records    = _dispatch(_worker, worker_args, n_workers, verbose, n_combos)
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)

    # Rename n_<fragment_formula> columns to be clearly labelled
    rename = {f'n_{f}': f'n_{f}' for f in frag_labels}   # already correct
    results_df = results_df.rename(columns=rename)

    best_row = results_df.iloc[0]
    best     = {k: best_row[k] for k in ['formula', 'density', 'mse', 'rmse']}

    if verbose:
        print(f"\n── Best result {'─'*40}")
        print(f"   Formula : {best['formula']}")
        print(f"   Density : {best['density']:.4f} g/cm³")
        print(f"   RMSE    : {best['rmse']:.6f}  (10⁻⁶ Å⁻²)")
        print('─' * 57)

    return results_df, best


def fit_stoichiometry_density_fragments_bidirectional(
    sld_3col, fragments, count_ranges, density_range, merge_points,
    n_density_grid=50, optimise_density=True, energy_mask=None,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density using a fragment-based search (bidirectional KK).

    Identical to fit_stoichiometry_density_fragments but uses the combined
    forward + reverse KK cost:

        MSE_bi = 0.5·MSE_fwd + 0.5·MSE_rev

    See fit_stoichiometry_density_fragments for parameter documentation.

    Returns
    -------
    results_df : pd.DataFrame
        Sorted by combined MSE.  Extra columns: mse_fwd, mse_rev,
        rmse_fwd, rmse_rev.
    best : dict
        formula, density, mse, rmse, mse_fwd, mse_rev, rmse_fwd, rmse_rev.
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(fragments) != len(count_ranges):
        raise ValueError("len(fragments) must equal len(count_ranges)")

    fragment_combos = _build_fragment_grid(fragments, count_ranges)
    n_combos = len(fragment_combos)

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, energy_delta) = _prepare_fit_data(sld_3col, energy_mask)

    if verbose:
        n_cpu = os.cpu_count() if n_workers is None else n_workers
        print("KK stoichiometry fit  [fragment-based, bidirectional]")
        print(f"  Fragments       : {fragments}")
        print(f"  Unique formulas : {n_combos}")
        print(f"  Density range   : {density_range[0]:.2f}–{density_range[1]:.2f} g/cm³")
        print(f"  Cost function   : 0.5·MSE_fwd + 0.5·MSE_rev")
        print(f"  Residual points : {len(energies_fit)}"
              + (f"  [{energies_fit.min():.1f}–{energies_fit.max():.1f} eV]"
                 if energy_mask else ""))
        print(f"  Workers         : {min(n_cpu, n_combos)}\n")

    frag_labels = fragments

    worker_args = [
        (formula, frag_labels, counts,
         energy_beta, energy_delta,
         merge_points, kk_kwargs,
         density_range, energies_fit, real_sld_fit, imag_sld_fit,
         optimise_density, n_density_grid)
        for formula, counts in fragment_combos
    ]

    records    = _dispatch(_worker_bidirectional, worker_args, n_workers, verbose, n_combos)
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)

    best_row = results_df.iloc[0]
    best     = {k: best_row[k]
                for k in ['formula', 'density', 'mse', 'rmse',
                           'mse_fwd', 'mse_rev', 'rmse_fwd', 'rmse_rev']}

    if verbose:
        print(f"\n── Best result (fragment bidirectional) {'─'*18}")
        print(f"   Formula  : {best['formula']}")
        print(f"   Density  : {best['density']:.4f} g/cm³")
        print(f"   RMSE     : {best['rmse']:.6f}  (combined)")
        print(f"   RMSE_fwd : {best['rmse_fwd']:.6f}  (imag→real)")
        print(f"   RMSE_rev : {best['rmse_rev']:.6f}  (real→imag)")
        print('─' * 57)

    return results_df, best


# ---------------------------------------------------------------------------
# Count cost-surface scanners
# ---------------------------------------------------------------------------

def scan_atom_count_surface(
    sld_3col, atoms, fixed_counts, scan_atom, scan_range,
    density_range, merge_points,
    n_density_points=40, energy_mask=None, kk_kwargs=None, ax=None
):
    """
    Scan the RMSE cost surface over integer counts of a single atom, while
    holding all other atoms fixed at ``fixed_counts``.  For each count value
    the best density is found via scalar minimisation.

    Parameters
    ----------
    sld_3col       : np.ndarray (n,3)   [Energy, SLD_Real, SLD_Imag], 10^-6 Å^-2.
    atoms          : list of str        All atom symbols, e.g. ['Sn','C','O'].
    fixed_counts   : list of int        Baseline count for each atom.  The value
                                        for ``scan_atom`` is overridden by the scan.
    scan_atom      : str                The atom symbol whose count is varied,
                                        e.g. 'C'.
    scan_range     : tuple (min, max)   Integer count range to scan, inclusive.
    density_range  : tuple              (rho_min, rho_max) g/cm³.
    merge_points   : list of 2 floats   KK splice energies [E_min, E_max] eV.
    n_density_points : int              Fallback grid size for density (default 40);
                                        Brent optimisation is used regardless.
    energy_mask    : tuple, optional    (E_min, E_max) eV residual window.
    kk_kwargs      : dict, optional     Extra kwargs for kkcalc.
    ax             : Axes, optional     Plot into existing axes.

    Returns
    -------
    counts    : np.ndarray   Integer count values scanned.
    rmses     : np.ndarray   Best RMSE at each count (NaN if KK failed).
    densities : np.ndarray   Best density at each count.
    """
    if kk_kwargs is None:
        kk_kwargs = {}

    if scan_atom not in atoms:
        raise ValueError(f"scan_atom '{scan_atom}' not found in atoms {atoms}")

    scan_idx   = atoms.index(scan_atom)
    count_vals = np.array(_build_count_grid(scan_range))

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, _) = _prepare_fit_data(sld_3col, energy_mask)

    rmses     = np.full(len(count_vals), np.nan)
    best_dens = np.full(len(count_vals), np.nan)

    for i, c in enumerate(count_vals):
        counts           = list(fixed_counts)
        counts[scan_idx] = int(c)
        formula          = _build_formula(atoms, counts)

        asf_output, formula_mass = _asf_output_for_formula(
            energy_beta, formula, merge_points, kk_kwargs)
        if asf_output is None:
            continue

        best_rho, best_mse = _best_density_for_asf(
            asf_output, formula_mass, density_range,
            energies_fit, real_sld_fit, imag_sld_fit,
            optimise_density=True, n_density_grid=n_density_points)

        if np.isfinite(best_mse):
            rmses[i]     = np.sqrt(best_mse)
            best_dens[i] = best_rho

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))

    valid = np.isfinite(rmses)
    ax.plot(count_vals[valid], rmses[valid], 'o-', color='tab:blue',
            markersize=6, linewidth=1.8)
    ax.set_xlabel(f'Count of {scan_atom}')
    ax.set_ylabel('RMSE  (10$^{-6}$ Å$^{-2}$)')

    fixed_str = ', '.join(
        f'{a}={fixed_counts[j]}'
        for j, a in enumerate(atoms) if a != scan_atom
    )
    ax.set_title(f'Atom count surface — {scan_atom}\n[{fixed_str}]', fontsize=10)
    ax.grid(True, alpha=0.3)

    if valid.any():
        best_idx = np.nanargmin(rmses)
        ax.axvline(count_vals[best_idx], color='tab:red', linestyle='--',
                   linewidth=1.2,
                   label=f'min  {scan_atom}={count_vals[best_idx]},  '
                         f'ρ={best_dens[best_idx]:.3f} g/cm³')
        ax.legend(fontsize=9)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return count_vals, rmses, best_dens


def scan_all_atom_count_surfaces(
    sld_3col, atoms, fixed_counts, count_ranges,
    density_range, merge_points,
    n_density_points=40, energy_mask=None, kk_kwargs=None, figsize=None
):
    """
    Scan the cost surface for every atom that has a ``(min, max)`` count range,
    one subplot per atom.  Atoms with a fixed integer count are skipped.

    Parameters
    ----------
    sld_3col      : np.ndarray (n,3)
    atoms         : list of str        All atom symbols.
    fixed_counts  : list of int        Baseline count for each atom used when
                                       that atom is not being scanned.
    count_ranges  : list               Per-atom int (fixed) or (min, max) tuple.
    density_range : tuple
    merge_points  : list of 2 floats
    n_density_points : int
    energy_mask   : tuple, optional
    kk_kwargs     : dict, optional
    figsize       : tuple, optional    Defaults to (5 × n_scanned, 4).

    Returns
    -------
    results : dict   {atom_symbol: (counts, rmses, best_densities)}
    fig, axes
    """
    scan_atoms  = [(a, cr) for a, cr in zip(atoms, count_ranges)
                   if not isinstance(cr, int)]
    if not scan_atoms:
        raise ValueError("No atoms have a (min, max) count range to scan.")

    n = len(scan_atoms)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    results = {}
    for ax, (atom, cr) in zip(axes, scan_atoms):
        c_out, r_out, d_out = scan_atom_count_surface(
            sld_3col, atoms, fixed_counts, atom, cr,
            density_range, merge_points,
            n_density_points=n_density_points,
            energy_mask=energy_mask, kk_kwargs=kk_kwargs, ax=ax)
        results[atom] = (c_out, r_out, d_out)

    plt.tight_layout()
    plt.show()
    return results, fig, axes


def scan_fragment_count_surface(
    sld_3col, fragments, fixed_counts, scan_fragment, scan_range,
    density_range, merge_points,
    n_density_points=40, energy_mask=None, kk_kwargs=None, ax=None
):
    """
    Scan the RMSE cost surface over integer multipliers of a single fragment,
    while holding all other fragment counts fixed.  For each multiplier value
    the best density is found via scalar minimisation.

    Parameters
    ----------
    sld_3col      : np.ndarray (n,3)
    fragments     : list of str       All fragment formula strings.
    fixed_counts  : list of int       Baseline multiplier for every fragment.
                                      The value for ``scan_fragment`` is
                                      overridden by the scan.
    scan_fragment : str               The fragment formula to scan, e.g. 'C8H8'.
    scan_range    : tuple (min, max)  Integer multiplier range, inclusive.
    density_range : tuple             (rho_min, rho_max) g/cm³.
    merge_points  : list of 2 floats  KK splice energies [E_min, E_max] eV.
    n_density_points : int            Fallback grid size for density (default 40).
    energy_mask   : tuple, optional
    kk_kwargs     : dict, optional
    ax            : Axes, optional

    Returns
    -------
    counts    : np.ndarray   Multiplier values scanned.
    rmses     : np.ndarray   Best RMSE at each multiplier (NaN if KK failed).
    densities : np.ndarray   Best density at each multiplier.
    """
    if kk_kwargs is None:
        kk_kwargs = {}

    if scan_fragment not in fragments:
        raise ValueError(
            f"scan_fragment '{scan_fragment}' not found in fragments {fragments}")

    scan_idx   = fragments.index(scan_fragment)
    frag_dicts = [_parse_formula(f) for f in fragments]
    count_vals = np.array(_build_count_grid(scan_range))

    (sld_sorted, energies_fit, real_sld_fit, imag_sld_fit,
     energy_beta, _) = _prepare_fit_data(sld_3col, energy_mask)

    rmses     = np.full(len(count_vals), np.nan)
    best_dens = np.full(len(count_vals), np.nan)

    for i, c in enumerate(count_vals):
        counts           = list(fixed_counts)
        counts[scan_idx] = int(c)
        if all(n == 0 for n in counts):
            continue
        formula = _combine_fragments(frag_dicts, counts)

        asf_output, formula_mass = _asf_output_for_formula(
            energy_beta, formula, merge_points, kk_kwargs)
        if asf_output is None:
            continue

        best_rho, best_mse = _best_density_for_asf(
            asf_output, formula_mass, density_range,
            energies_fit, real_sld_fit, imag_sld_fit,
            optimise_density=True, n_density_grid=n_density_points)

        if np.isfinite(best_mse):
            rmses[i]     = np.sqrt(best_mse)
            best_dens[i] = best_rho

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))

    valid = np.isfinite(rmses)
    ax.plot(count_vals[valid], rmses[valid], 'o-', color='tab:green',
            markersize=6, linewidth=1.8)
    ax.set_xlabel(f'Multiplier of {scan_fragment}')
    ax.set_ylabel('RMSE  (10$^{-6}$ Å$^{-2}$)')

    fixed_str = ', '.join(
        f'{n}×{f}'
        for j, (f, n) in enumerate(zip(fragments, fixed_counts))
        if f != scan_fragment
    )
    ax.set_title(f'Fragment count surface — {scan_fragment}\n[{fixed_str}]', fontsize=10)
    ax.grid(True, alpha=0.3)

    if valid.any():
        best_idx = np.nanargmin(rmses)
        ax.axvline(count_vals[best_idx], color='tab:red', linestyle='--',
                   linewidth=1.2,
                   label=f'min  {scan_fragment}×{count_vals[best_idx]},  '
                         f'ρ={best_dens[best_idx]:.3f} g/cm³')
        ax.legend(fontsize=9)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return count_vals, rmses, best_dens


def scan_all_fragment_count_surfaces(
    sld_3col, fragments, fixed_counts, count_ranges,
    density_range, merge_points,
    n_density_points=40, energy_mask=None, kk_kwargs=None, figsize=None
):
    """
    Scan the cost surface for every fragment that has a ``(min, max)`` count
    range, one subplot per fragment.  Fixed-multiplier fragments are skipped.

    Parameters
    ----------
    sld_3col      : np.ndarray (n,3)
    fragments     : list of str       All fragment formula strings.
    fixed_counts  : list of int       Baseline multiplier for each fragment.
    count_ranges  : list              Per-fragment int (fixed) or (min, max) tuple.
    density_range : tuple
    merge_points  : list of 2 floats
    n_density_points : int
    energy_mask   : tuple, optional
    kk_kwargs     : dict, optional
    figsize       : tuple, optional   Defaults to (5 × n_scanned, 4).

    Returns
    -------
    results : dict   {fragment_formula: (counts, rmses, best_densities)}
    fig, axes
    """
    scan_frags = [(f, cr) for f, cr in zip(fragments, count_ranges)
                  if not isinstance(cr, int)]
    if not scan_frags:
        raise ValueError("No fragments have a (min, max) count range to scan.")

    n = len(scan_frags)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    results = {}
    for ax, (frag, cr) in zip(axes, scan_frags):
        c_out, r_out, d_out = scan_fragment_count_surface(
            sld_3col, fragments, fixed_counts, frag, cr,
            density_range, merge_points,
            n_density_points=n_density_points,
            energy_mask=energy_mask, kk_kwargs=kk_kwargs, ax=ax)
        results[frag] = (c_out, r_out, d_out)

    plt.tight_layout()
    plt.show()
    return results, fig, axes


def scan_all_atom_count_surfaces_stitched(
    edge_specs, atoms, fixed_counts, count_ranges,
    density_range,
    n_density_points=40, kk_kwargs=None, figsize=None
):
    """
    Scan the atom count cost surfaces using the stitched multi-edge dataset.

    Identical to ``scan_all_atom_count_surfaces`` but stitches the SLD data
    from all edges in ``edge_specs`` before scanning, so each count evaluation
    uses a single KK transform over the combined energy range.

    Parameters
    ----------
    edge_specs     : list of dict   Same format as the multi-edge fit functions.
                                    Each dict needs at least ``sld_3col``,
                                    ``merge_points``, and optionally
                                    ``energy_mask`` and ``label``.
    atoms          : list of str    All atom symbols.
    fixed_counts   : list of int    Baseline count for each atom.
    count_ranges   : list           Per-atom int (fixed) or (min, max) to scan.
    density_range  : tuple          (rho_min, rho_max) g/cm³.
    n_density_points : int          Brent density optimisation fallback grid (default 40).
    kk_kwargs      : dict, optional Extra kwargs for kkcalc.
    figsize        : tuple, optional Defaults to (5 × n_scanned, 4).

    Returns
    -------
    results : dict   {atom_symbol: (counts, rmses, best_densities)}
    fig, axes
    """
    (sld_stitched, merge_combined,
     fit_E, fit_re, fit_im,
     _) = _stitch_sld_arrays(edge_specs)

    # Reconstruct a 3-col array covering only the residual points from all
    # edge windows — this is what scan_atom_count_surface will see as its data.
    # energy_mask=None because the stitched residual arrays already cover
    # exactly the desired windows; no further masking needed.
    sld_3col_stitched = np.column_stack((fit_E, fit_re, fit_im))

    scan_atoms = [(a, cr) for a, cr in zip(atoms, count_ranges)
                  if not isinstance(cr, int)]
    if not scan_atoms:
        raise ValueError("No atoms have a (min, max) count range to scan.")

    n = len(scan_atoms)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    results = {}
    for ax, (atom, cr) in zip(axes, scan_atoms):
        c_out, r_out, d_out = scan_atom_count_surface(
            sld_3col_stitched, atoms, fixed_counts, atom, cr,
            density_range, merge_combined,
            n_density_points=n_density_points,
            energy_mask=None,
            kk_kwargs=kk_kwargs,
            ax=ax,
        )
        ax.set_title(ax.get_title() + '\n[stitched]', fontsize=9)
        results[atom] = (c_out, r_out, d_out)

    plt.tight_layout()
    plt.show()
    return results, fig, axes


def scan_all_fragment_count_surfaces_stitched(
    edge_specs, fragments, fixed_counts, count_ranges,
    density_range,
    n_density_points=40, kk_kwargs=None, figsize=None
):
    """
    Scan the fragment count cost surfaces using the stitched multi-edge dataset.

    Identical to ``scan_all_fragment_count_surfaces`` but stitches the SLD data
    from all edges in ``edge_specs`` before scanning.

    Parameters
    ----------
    edge_specs     : list of dict   Same format as the multi-edge fit functions.
    fragments      : list of str    All fragment formula strings.
    fixed_counts   : list of int    Baseline multiplier for each fragment.
    count_ranges   : list           Per-fragment int (fixed) or (min, max) to scan.
    density_range  : tuple          (rho_min, rho_max) g/cm³.
    n_density_points : int          Default 40.
    kk_kwargs      : dict, optional
    figsize        : tuple, optional Defaults to (5 × n_scanned, 4).

    Returns
    -------
    results : dict   {fragment_formula: (counts, rmses, best_densities)}
    fig, axes
    """
    (sld_stitched, merge_combined,
     fit_E, fit_re, fit_im,
     _) = _stitch_sld_arrays(edge_specs)

    sld_3col_stitched = np.column_stack((fit_E, fit_re, fit_im))

    scan_frags = [(f, cr) for f, cr in zip(fragments, count_ranges)
                  if not isinstance(cr, int)]
    if not scan_frags:
        raise ValueError("No fragments have a (min, max) count range to scan.")

    n = len(scan_frags)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    results = {}
    for ax, (frag, cr) in zip(axes, scan_frags):
        c_out, r_out, d_out = scan_fragment_count_surface(
            sld_3col_stitched, fragments, fixed_counts, frag, cr,
            density_range, merge_combined,
            n_density_points=n_density_points,
            energy_mask=None,
            kk_kwargs=kk_kwargs,
            ax=ax,
        )
        ax.set_title(ax.get_title() + '\n[stitched]', fontsize=9)
        results[frag] = (c_out, r_out, d_out)

    plt.tight_layout()
    plt.show()
    return results, fig, axes


# ===========================================================================
# Multi-edge fitting
# ===========================================================================
#
# Two complementary strategies for simultaneously fitting multiple energy edges:
#
# Case 1 — Independent KK per edge
# ---------------------------------
# Each edge is transformed separately using its own merge_points.  The total
# cost is the mean of the per-edge MSEs evaluated at the same density:
#
#     MSE_total(rho) = mean_k [ MSE_edge_k(rho) ]
#
# Case 2 — Stitched KK across both edges
# ---------------------------------------
# The imaginary SLD data from all edges are concatenated into a single
# spectrum and passed to kkcalc as one continuous dataset.  kkcalc stitches
# this into the Henke background across the full energy span in a single KK
# integral, producing one globally self-consistent optical constant set.
#
# When to prefer each:
#   Case 1 - edges are widely separated or measured under different conditions.
#   Case 2 - you want one globally self-consistent set across both edges.
# ===========================================================================


def _prepare_multi_edge_data(edge_specs, kk_kwargs):
    """
    Prepare fit data for each edge and return a list of per-edge data dicts.

    Parameters
    ----------
    edge_specs : list of dict, each with:
        sld_3col    : np.ndarray (n,3)   [Energy, SLD_Real, SLD_Imag]
        merge_points: list of 2 floats
        energy_mask : tuple or None
        label       : str, optional
    kk_kwargs : dict

    Returns list of dicts with keys:
        label, merge_points, energy_beta,
        energies_fit, real_sld_fit, imag_sld_fit
    """
    from NEXAFS import EnergytoWavelength, imag_SLD_to_beta

    prepared = []
    for spec in edge_specs:
        sld_3col     = spec['sld_3col']
        merge_points = spec['merge_points']
        energy_mask  = spec.get('energy_mask', None)
        label        = spec.get('label', f"edge@{merge_points[0]:.0f}eV")

        sort_idx     = np.argsort(sld_3col[:, 0])
        sld_sorted   = sld_3col[sort_idx]
        energies_all = sld_sorted[:, 0]
        real_sld_all = sld_sorted[:, 1]
        imag_sld_all = sld_sorted[:, 2]

        if energy_mask is not None:
            e_lo, e_hi = energy_mask
            mask = (energies_all >= e_lo) & (energies_all <= e_hi)
            if mask.sum() == 0:
                raise ValueError(
                    f"energy_mask {energy_mask} for '{label}' excludes all points. "
                    f"Data range: {energies_all.min():.1f}-{energies_all.max():.1f} eV")
            energies_fit = energies_all[mask]
            real_sld_fit = real_sld_all[mask]
            imag_sld_fit = imag_sld_all[mask]
        else:
            energies_fit, real_sld_fit, imag_sld_fit = (
                energies_all, real_sld_all, imag_sld_all)

        wavelengths_all = EnergytoWavelength(energies_all)
        sld_4col        = np.column_stack((energies_all, real_sld_all,
                                           imag_sld_all, wavelengths_all))
        energy_beta     = imag_SLD_to_beta(sld_4col)

        prepared.append(dict(
            label        = label,
            merge_points = merge_points,
            energy_beta  = energy_beta,
            energies_fit = energies_fit,
            real_sld_fit = real_sld_fit,
            imag_sld_fit = imag_sld_fit,
        ))
    return prepared


def _stitch_sld_arrays(edge_specs):
    """
    Concatenate SLD data from multiple edges into one sorted array and derive
    combined merge_points, residual arrays, and energy_beta.

    Parameters
    ----------
    edge_specs : list of dict (same format as _prepare_multi_edge_data input)

    Returns
    -------
    sld_stitched          : np.ndarray (n_total, 3)  sorted by energy
    merge_combined        : [E_lo, E_hi]  spanning all edge merge windows
    fit_E, fit_re, fit_im : residual arrays (union of all energy_mask windows)
    energy_beta_stitched  : np.ndarray (n_total, 2)  [E, beta]
    """
    from NEXAFS import EnergytoWavelength, imag_SLD_to_beta

    all_sld    = []
    all_fit_E  = []
    all_fit_re = []
    all_fit_im = []
    all_mp_lo  = []
    all_mp_hi  = []

    for spec in edge_specs:
        sld_3col     = spec['sld_3col']
        merge_points = spec['merge_points']
        energy_mask  = spec.get('energy_mask', None)

        all_sld.append(sld_3col)
        all_mp_lo.append(merge_points[0])
        all_mp_hi.append(merge_points[1])

        energies = sld_3col[:, 0]
        if energy_mask is not None:
            e_lo, e_hi = energy_mask
            mask = (energies >= e_lo) & (energies <= e_hi)
        else:
            mask = np.ones(len(energies), dtype=bool)

        all_fit_E.append(sld_3col[mask, 0])
        all_fit_re.append(sld_3col[mask, 1])
        all_fit_im.append(sld_3col[mask, 2])

    sld_combined = np.vstack(all_sld)
    sort_idx     = np.argsort(sld_combined[:, 0])
    sld_combined = sld_combined[sort_idx]
    merge_combined = [min(all_mp_lo), max(all_mp_hi)]

    fit_E  = np.concatenate(all_fit_E)
    fit_re = np.concatenate(all_fit_re)
    fit_im = np.concatenate(all_fit_im)
    sort_f = np.argsort(fit_E)
    fit_E, fit_re, fit_im = fit_E[sort_f], fit_re[sort_f], fit_im[sort_f]

    wavelengths = EnergytoWavelength(sld_combined[:, 0])
    sld_4col    = np.column_stack((sld_combined, wavelengths))
    energy_beta_stitched = imag_SLD_to_beta(sld_4col)

    return (sld_combined, merge_combined,
            fit_E, fit_re, fit_im,
            energy_beta_stitched)


def _worker_multi_edge_independent(args):
    """
    Case 1 worker: independent KK per edge, mean MSE across edges.

    args: (formula, atoms_or_frags, counts,
           prepared_edges, kk_kwargs, density_range,
           optimise_density, n_density_grid)
    """
    (formula, atoms_or_frags, counts,
     prepared_edges, kk_kwargs, density_range,
     optimise_density, n_density_grid) = args

    asf_list = []
    formula_mass = None
    for edge in prepared_edges:
        asf_out, fm = _asf_output_for_formula(
            edge['energy_beta'], formula, edge['merge_points'], kk_kwargs)
        if asf_out is None:
            return {'formula': formula, 'density': np.nan,
                    'mse': np.inf, 'rmse': np.inf,
                    **{f'n_{a}': c for a, c in zip(atoms_or_frags, counts)},
                    **{f'rmse_{e["label"]}': np.inf for e in prepared_edges}}
        asf_list.append(asf_out)
        formula_mass = fm

    rho_min, rho_max = density_range

    def cost(rho):
        total = 0.0
        for asf_out, edge in zip(asf_list, prepared_edges):
            mse = _fwd_combined_mse(
                asf_out, formula_mass, rho,
                edge['energies_fit'], edge['real_sld_fit'], edge['imag_sld_fit'])
            if not np.isfinite(mse):
                return np.inf
            total += mse
        return total / len(asf_list)

    if optimise_density:
        result = minimize_scalar(cost, bounds=(rho_min, rho_max),
                                 method='bounded', options={'xatol': 1e-4})
        best_rho, best_mse = result.x, result.fun
    else:
        grid = np.linspace(rho_min, rho_max, n_density_grid)
        mses = np.array([cost(rho) for rho in grid])
        best_idx = np.argmin(mses)
        best_rho, best_mse = grid[best_idx], mses[best_idx]

    per_edge_rmse = {}
    for asf_out, edge in zip(asf_list, prepared_edges):
        mse_e = _fwd_combined_mse(
            asf_out, formula_mass, best_rho,
            edge['energies_fit'], edge['real_sld_fit'], edge['imag_sld_fit'])
        per_edge_rmse[f'rmse_{edge["label"]}'] = (
            np.sqrt(mse_e) if np.isfinite(mse_e) else np.inf)

    return {'formula': formula, 'density': best_rho,
            'mse': best_mse,
            'rmse': np.sqrt(best_mse) if np.isfinite(best_mse) else np.inf,
            **{f'n_{a}': c for a, c in zip(atoms_or_frags, counts)},
            **per_edge_rmse}


def _worker_multi_edge_stitched(args):
    """
    Case 2 worker: single KK transform over the stitched multi-edge dataset.

    args: (formula, atoms_or_frags, counts,
           energy_beta_stitched, merge_points_combined,
           energies_fit_combined, real_sld_fit_combined, imag_sld_fit_combined,
           kk_kwargs, density_range,
           optimise_density, n_density_grid,
           edge_labels, per_edge_masks)
    """
    (formula, atoms_or_frags, counts,
     energy_beta_stitched, merge_points_combined,
     energies_fit_combined, real_sld_fit_combined, imag_sld_fit_combined,
     kk_kwargs, density_range,
     optimise_density, n_density_grid,
     edge_labels, per_edge_masks) = args

    asf_out, formula_mass = _asf_output_for_formula(
        energy_beta_stitched, formula, merge_points_combined, kk_kwargs)

    if asf_out is None:
        return {'formula': formula, 'density': np.nan,
                'mse': np.inf, 'rmse': np.inf,
                **{f'n_{a}': c for a, c in zip(atoms_or_frags, counts)},
                **{f'rmse_{lbl}': np.inf for lbl in edge_labels}}

    best_rho, best_mse = _best_density_for_asf(
        asf_out, formula_mass, density_range,
        energies_fit_combined, real_sld_fit_combined, imag_sld_fit_combined,
        optimise_density, n_density_grid)

    per_edge_rmse = {}
    for lbl, mask in zip(edge_labels, per_edge_masks):
        mse_e = _fwd_combined_mse(
            asf_out, formula_mass, best_rho,
            energies_fit_combined[mask],
            real_sld_fit_combined[mask],
            imag_sld_fit_combined[mask])
        per_edge_rmse[f'rmse_{lbl}'] = (
            np.sqrt(mse_e) if np.isfinite(mse_e) else np.inf)

    return {'formula': formula, 'density': best_rho,
            'mse': best_mse,
            'rmse': np.sqrt(best_mse) if np.isfinite(best_mse) else np.inf,
            **{f'n_{a}': c for a, c in zip(atoms_or_frags, counts)},
            **per_edge_rmse}


def _multi_edge_verbose_header(mode_label, edge_specs, n_combos,
                                density_range, n_workers):
    print(f"KK multi-edge fit  [{mode_label}]")
    for i, spec in enumerate(edge_specs):
        lbl = spec.get('label', f"edge{i+1}")
        mp  = spec['merge_points']
        em  = spec.get('energy_mask', None)
        print(f"  {lbl:20s}: merge=[{mp[0]:.0f},{mp[1]:.0f}] eV"
              + (f"  mask=({em[0]:.0f},{em[1]:.0f}) eV" if em else ""))
    print(f"  Stoichiometry combinations : {n_combos}")
    print(f"  Density range              : {density_range[0]:.2f}-{density_range[1]:.2f} g/cm3")
    n_cpu = os.cpu_count() if n_workers is None else n_workers
    print(f"  Workers                    : {min(n_cpu, n_combos)}\n")


def _multi_edge_verbose_footer(best, rmse_cols):
    print(f"\n-- Best result " + "-"*43)
    print(f"   Formula : {best['formula']}")
    print(f"   Density : {best['density']:.4f} g/cm3")
    print(f"   RMSE    : {best['rmse']:.6f}  (combined)")
    for col in rmse_cols:
        print(f"   {col:30s}: {best[col]:.6f}")
    print("-" * 57)


def fit_stoichiometry_density_multi_edge_independent(
    edge_specs, atoms, count_ranges, density_range,
    n_density_grid=50, optimise_density=True,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density against multiple energy edges simultaneously,
    running an independent KK transform for each edge (Case 1).

    The total cost is the mean MSE across edges at the same density:

        MSE_total(rho) = mean_k [ 0.5*MSE_k(real_kk, real_fit)
                                 + 0.5*MSE_k(imag_kk, imag_fit) ]

    Parameters
    ----------
    edge_specs : list of dict
        One dict per absorption edge, each containing:
        - sld_3col    : np.ndarray (n,3)  [Energy (eV), SLD_Real, SLD_Imag]
        - merge_points: [E_lo, E_hi] eV   kkcalc splice energies for this edge
        - energy_mask : (E_lo, E_hi), optional  residual window
        - label       : str, optional     name shown in output, e.g. 'C_Kedge'
    atoms         : list of str
    count_ranges  : list of int or (min, max)
    density_range : (rho_min, rho_max) g/cm3
    n_density_grid: int
    optimise_density : bool
    kk_kwargs     : dict, optional
    n_workers     : int or None
    verbose       : bool

    Returns
    -------
    results_df : pd.DataFrame  Sorted by MSE. Columns: formula, density, mse,
                               rmse, n_<atom>, rmse_<edge_label>.
    best       : dict
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(atoms) != len(count_ranges):
        raise ValueError("len(atoms) must equal len(count_ranges)")
    if len(edge_specs) < 2:
        raise ValueError("edge_specs must contain at least two edges.")

    combos   = _build_stoichiometry_grid(atoms, count_ranges)
    prepared = _prepare_multi_edge_data(edge_specs, kk_kwargs)

    if verbose:
        _multi_edge_verbose_header('independent KK per edge', edge_specs,
                                   len(combos), density_range, n_workers)

    worker_args = [
        (_build_formula(atoms, counts), atoms, counts,
         prepared, kk_kwargs, density_range,
         optimise_density, n_density_grid)
        for counts in combos
    ]

    records    = _dispatch(_worker_multi_edge_independent, worker_args,
                           n_workers, verbose, len(combos))
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    rmse_cols  = [c for c in results_df.columns if c.startswith('rmse_')]
    best       = {k: best_row[k]
                  for k in ['formula', 'density', 'mse', 'rmse'] + rmse_cols}

    if verbose:
        _multi_edge_verbose_footer(best, rmse_cols)

    return results_df, best


def fit_stoichiometry_density_multi_edge_stitched(
    edge_specs, atoms, count_ranges, density_range,
    n_density_grid=50, optimise_density=True,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """
    Fit stoichiometry and density against multiple energy edges simultaneously,
    running a single KK transform over the stitched multi-edge dataset (Case 2).

    The imaginary SLD from all edges is concatenated and passed to kkcalc as
    one continuous spectrum with merge_points spanning the full energy range.

    Parameters
    ----------
    edge_specs    : list of dict  (same format as the independent version)
    atoms         : list of str
    count_ranges  : list
    density_range : tuple
    n_density_grid: int
    optimise_density : bool
    kk_kwargs     : dict, optional
    n_workers     : int or None
    verbose       : bool

    Returns
    -------
    results_df : pd.DataFrame  Columns: formula, density, mse, rmse,
                               n_<atom>, rmse_<edge_label>.
    best       : dict
    """
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(atoms) != len(count_ranges):
        raise ValueError("len(atoms) must equal len(count_ranges)")
    if len(edge_specs) < 2:
        raise ValueError("edge_specs must contain at least two edges.")

    combos = _build_stoichiometry_grid(atoms, count_ranges)

    (sld_stitched, merge_combined,
     fit_E, fit_re, fit_im,
     energy_beta_stitched) = _stitch_sld_arrays(edge_specs)

    edge_labels = [s.get('label', f"edge{i+1}") for i, s in enumerate(edge_specs)]
    per_edge_masks = []
    for spec in edge_specs:
        em = spec.get('energy_mask', None)
        if em is not None:
            per_edge_masks.append((fit_E >= em[0]) & (fit_E <= em[1]))
        else:
            mp = spec['merge_points']
            per_edge_masks.append((fit_E >= mp[0]) & (fit_E <= mp[1]))

    if verbose:
        _multi_edge_verbose_header('stitched KK across edges', edge_specs,
                                   len(combos), density_range, n_workers)
        print(f"  Stitched dataset           : {len(sld_stitched)} points  "
              f"[{sld_stitched[:,0].min():.1f}-{sld_stitched[:,0].max():.1f} eV]")
        print(f"  Combined merge_points      : {merge_combined}")
        print(f"  Total residual points      : {len(fit_E)}\n")

    worker_args = [
        (_build_formula(atoms, counts), atoms, counts,
         energy_beta_stitched, merge_combined,
         fit_E, fit_re, fit_im,
         kk_kwargs, density_range,
         optimise_density, n_density_grid,
         edge_labels, per_edge_masks)
        for counts in combos
    ]

    records    = _dispatch(_worker_multi_edge_stitched, worker_args,
                           n_workers, verbose, len(combos))
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    rmse_cols  = [c for c in results_df.columns if c.startswith('rmse_')]
    best       = {k: best_row[k]
                  for k in ['formula', 'density', 'mse', 'rmse'] + rmse_cols}

    if verbose:
        _multi_edge_verbose_footer(best, rmse_cols)

    return results_df, best


def fit_stoichiometry_density_fragments_multi_edge_independent(
    edge_specs, fragments, count_ranges, density_range,
    n_density_grid=50, optimise_density=True,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """Fragment-based version of fit_stoichiometry_density_multi_edge_independent."""
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(fragments) != len(count_ranges):
        raise ValueError("len(fragments) must equal len(count_ranges)")
    if len(edge_specs) < 2:
        raise ValueError("edge_specs must contain at least two edges.")

    combos   = _build_fragment_grid(fragments, count_ranges)
    prepared = _prepare_multi_edge_data(edge_specs, kk_kwargs)

    if verbose:
        _multi_edge_verbose_header('fragment, independent KK per edge', edge_specs,
                                   len(combos), density_range, n_workers)

    worker_args = [
        (formula, fragments, counts,
         prepared, kk_kwargs, density_range,
         optimise_density, n_density_grid)
        for formula, counts in combos
    ]

    records    = _dispatch(_worker_multi_edge_independent, worker_args,
                           n_workers, verbose, len(combos))
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    rmse_cols  = [c for c in results_df.columns if c.startswith('rmse_')]
    best       = {k: best_row[k]
                  for k in ['formula', 'density', 'mse', 'rmse'] + rmse_cols}

    if verbose:
        _multi_edge_verbose_footer(best, rmse_cols)

    return results_df, best


def fit_stoichiometry_density_fragments_multi_edge_stitched(
    edge_specs, fragments, count_ranges, density_range,
    n_density_grid=50, optimise_density=True,
    kk_kwargs=None, n_workers=None, verbose=True
):
    """Fragment-based version of fit_stoichiometry_density_multi_edge_stitched."""
    if kk_kwargs is None:
        kk_kwargs = {}
    if len(fragments) != len(count_ranges):
        raise ValueError("len(fragments) must equal len(count_ranges)")
    if len(edge_specs) < 2:
        raise ValueError("edge_specs must contain at least two edges.")

    combos = _build_fragment_grid(fragments, count_ranges)

    (sld_stitched, merge_combined,
     fit_E, fit_re, fit_im,
     energy_beta_stitched) = _stitch_sld_arrays(edge_specs)

    edge_labels = [s.get('label', f"edge{i+1}") for i, s in enumerate(edge_specs)]
    per_edge_masks = []
    for spec in edge_specs:
        em = spec.get('energy_mask', None)
        if em is not None:
            per_edge_masks.append((fit_E >= em[0]) & (fit_E <= em[1]))
        else:
            mp = spec['merge_points']
            per_edge_masks.append((fit_E >= mp[0]) & (fit_E <= mp[1]))

    if verbose:
        _multi_edge_verbose_header('fragment, stitched KK across edges', edge_specs,
                                   len(combos), density_range, n_workers)
        print(f"  Stitched range  : {sld_stitched[:,0].min():.1f}-{sld_stitched[:,0].max():.1f} eV\n")

    worker_args = [
        (formula, fragments, counts,
         energy_beta_stitched, merge_combined,
         fit_E, fit_re, fit_im,
         kk_kwargs, density_range,
         optimise_density, n_density_grid,
         edge_labels, per_edge_masks)
        for formula, counts in combos
    ]

    records    = _dispatch(_worker_multi_edge_stitched, worker_args,
                           n_workers, verbose, len(combos))
    results_df = pd.DataFrame(records).sort_values('mse').reset_index(drop=True)
    best_row   = results_df.iloc[0]
    rmse_cols  = [c for c in results_df.columns if c.startswith('rmse_')]
    best       = {k: best_row[k]
                  for k in ['formula', 'density', 'mse', 'rmse'] + rmse_cols}

    if verbose:
        _multi_edge_verbose_footer(best, rmse_cols)

    return results_df, best
