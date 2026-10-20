"""
settings.py
-----------
Default thresholds for classification, selection, trimming and QC.

Every value is copied into a project's ``reduction.json`` at ``init`` and can be
changed there (``python -m rsoxr_reduce set key.sub value``).  Trim/QC values
marked CAL are tuned by ``calibrate`` against the manual Nov 2025 reductions;
see CALIBRATION below for when and how well.
"""

import copy

DEFAULTS = {
    "classify": {
        "min_bytes": 200,             # smaller files are start/stop fillers
        "dummy_max_points": 5,
        "dummy_max_angle": 1.0,       # 'd 0 -> 1 n=2' script lines
        "angle_max": 90.0,
        "energy_min": 100.0,          # column 0 above this is an energy (eV)
        "energy_scan_min_span": 5.0,
    },
    "group": {
        "energy_tol": 0.05,           # eV; set-point jitter < 0.004, grids >= 0.2
        "position_tol": 0.2,          # mm between sections of one energy -> POSITION_MISMATCH
        "position_review_mm": 3.0,    # ... held for review only beyond this
        "section_match_deg": 0.6,     # start/end angle tolerance for a section
        "section_min_support": 0.25,  # fraction of energies a section must appear in
    },
    "select": {
        "partial_frac": 0.8,          # span < frac * nominal span -> PARTIAL_RANGE
        "off_date_hours": 6.0,
    },
    "trim": {
        # PD start (beam clipping), per folder
        "clip_search_deg": {"C": 1.5, "O": 1.2, "default": 1.5},
        "clip_shift_steps": 2,        # own peak this many steps off -> CLIP_SHIFT
        "min_start": 1,               # never keep the θ≈0 direct-beam point
        "pd_sat_level": 9.99,         # V; the PD ADC saturates at 10.0005 V
        # PD background (dark offset), per folder
        "bg_fit_min_counts": 200.0,   # CEM counts used in the PD-vs-CT regression
        "bg_fit_max_sigma": 0.003,    # reject noisy regressions
        "bg_fit_min_energies": 3,
        "bg_default": {"C": 0.0024, "O": 0.0020, "default": 0.0022},  # Nov 2025 medians
        "pd_sigma_default": 0.0008,   # PD noise (ADC step 3.05e-4)
        # junctions / tails
        "scale_rel_target": 0.02,     # stop overlap once the scale is known to 2 %
        "min_overlap": 4,             # points
        "snr_min": 5.0,               # PD usable while smoothed (PD-bg) >= snr_min*σ
        "floor_outlier_ratio": 0.33,  # PD kept past its SNR floor for the overlap: dropouts
                                      # (PD-bg) < this × 5-pt running median are not written
                                      # (nor PD points the next section covers); see trim.py
        "ct_min_counts": 10.0,        # CEM usable while smoothed counts >= this
        "ct_max_rate": 1.0e5,         # cps; CEM non-linear above (Nov 2025 data)
        "ct_end_min": 1,              # always drop the final point (legacy convention)
    },
    "open_beam": {
        "slit_tol": 0.03,             # relative exit-slit gap difference allowed
        "gain_correct": True,         # I0 × 10^(gain_sample - gain_ob); False = legacy
    },
    "qc": {
        "noisy_overlap": 0.10,        # junction scale uncertainty -> NOISY_OVERLAP (info)
        "jump": 0.20,                 # ratio drift across an overlap -> JUMP
        "scale_outlier_mad": 3.0,     # k in |dev| > k·σ_tot
        "scale_outlier_min": 0.30,    # ... and > 30 % (gross errors; detector
                                      # sensitivity has real structure at edges)
        "trend_window": 3,            # neighbouring energies each side
        "trend_max_de": 6.0,          # eV; only nearby energies define the trend
        "trend_sigma_min": 0.05,      # floor on the trend's own uncertainty (log)
        "blend_noisy_scales": True,   # combine noisy overlap scales with the trend
        "use_trend_for_no_overlap": True,
        "gap_points": 4,              # points each side for gap bridging
        "max_gap_deg": 0.5,
        "low_counts_tail_frac": 0.2,
        "fringe_damping": 0.25,       # upper/lower contrast below 0.75 -> FRINGE_DAMPING
        "fringe_min_contrast": 0.03,  # rms log-R fringe contrast needed to judge
        "fringe_min_corr": 0.5,
        "fringe_window_deg": 6.0,     # adjacent-window fallback when the overlap is short       # residual correlation below -> FRINGE_MISMATCH
    },
    "output": {
        "dir": "ReducedAuto",
        "error_frac": 0.01,           # flat 1 % error, as the widget wrote
        "error_model": "flat",        # "flat" | "counting": add PD σ/(PD-bg) or CEM
                                      # 1/√counts (smoothed signal) in quadrature
        "name": "{sample}_{energy:.1f}eV.dat",
    },
}

# Filled in by calibrate (date, sample set, agreement scores).
CALIBRATION = {}


def defaults():
    return copy.deepcopy(DEFAULTS)


def edge_value(d, edge):
    """Per-edge dict lookup with a 'default' key."""
    if isinstance(d, dict):
        return d.get(edge, d.get("default"))
    return d
