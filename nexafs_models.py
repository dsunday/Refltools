"""NEXAFS peak-model seeds/bounds extracted from LARIAT/BrewerNEXAFS_Fits.ipynb.

Generated once from the notebook cells (index in comment); edit here, not in the notebook.
Each model = dict(peaks=..., edge=..., baseline=0). Peak bounds are (lo, hi, fit_flag).
"""

# notebook cell 27
SOC_CARBON_BAKE = dict(
    peaks={'peak_1_energy': 284.9,
 'peak_1_energy_bounds': (284.9, 285.1, False),
 'peak_1_width': 0.6,
 'peak_1_width_bounds': (0.4, 0.8, True),
 'peak_1_height': 0.3,
 'peak_1_height_bounds': (0.1, 0.3, True),
 'peak_2_energy': 285.5,
 'peak_2_energy_bounds': (285.4, 285.6, False),
 'peak_2_width': 0.6,
 'peak_2_width_bounds': (0.4, 1, True),
 'peak_2_height': 2.6,
 'peak_2_height_bounds': (1.8, 4, False),
 'peak_3_energy': 286.7,
 'peak_3_energy_bounds': (286.6, 286.9, True),
 'peak_3_width': 0.6,
 'peak_3_width_bounds': (0.3, 0.8, True),
 'peak_3_height': 0.5,
 'peak_3_height_bounds': (0.5, 1.3, True),
 'peak_4_energy': 287.5,
 'peak_4_energy_bounds': (287.4, 287.6, False),
 'peak_4_width': 1.5,
 'peak_4_width_bounds': (0.8, 1.5, True),
 'peak_4_height': 0.8,
 'peak_4_height_bounds': (0.2, 0.8, True),
 'peak_5_energy': 288.9,
 'peak_5_energy_bounds': (288.9, 289.1, True),
 'peak_5_width': 1.2,
 'peak_5_width_bounds': (0.8, 1.2, True),
 'peak_5_height': 2,
 'peak_5_height_bounds': (1.5, 2.9, True),
 'peak_6_energy': 290.4,
 'peak_6_energy_bounds': (290.2, 290.6, True),
 'peak_6_width': 0.8,
 'peak_6_width_bounds': (0.8, 1.2, True),
 'peak_6_height': 0.2,
 'peak_6_height_bounds': (0.2, 0.8, True),
 'peak_7_energy': 293.1,
 'peak_7_energy_bounds': (292.9, 293.3, True),
 'peak_7_width': 4.7,
 'peak_7_width_bounds': (3.2, 4.8, True),
 'peak_7_height': 0.6,
 'peak_7_height_bounds': (0.2, 0.8, True),
 'peak_8_energy': 296.9,
 'peak_8_energy_bounds': (295.9, 297, True),
 'peak_8_width': 4.8,
 'peak_8_width_bounds': (3.2, 4.8, True),
 'peak_8_height': 0.5,
 'peak_8_height_bounds': (0.2, 0.8, True),
 'peak_9_energy': 303,
 'peak_9_energy_bounds': (302.9, 303.1, False),
 'peak_9_width': 8.4,
 'peak_9_width_bounds': (6.4, 9.6, True),
 'peak_9_height': 0.5,
 'peak_9_height_bounds': (0.2, 0.8, True)},
    edge={'location': 289.5, 'height': 1.43, 'width': 0.6, 'decay': 0.01},
    baseline=0,
)

# notebook cell 30
SOC_CARBON_UV = dict(
    peaks={'peak_1_energy': 284.9,
 'peak_1_energy_bounds': (284.9, 285.1, False),
 'peak_1_width': 0.6,
 'peak_1_width_bounds': (0.4, 0.8, True),
 'peak_1_height': 0.3,
 'peak_1_height_bounds': (0.1, 0.3, True),
 'peak_2_energy': 285.5,
 'peak_2_energy_bounds': (285.4, 285.6, False),
 'peak_2_width': 0.6,
 'peak_2_width_bounds': (0.4, 1, True),
 'peak_2_height': 1.7,
 'peak_2_height_bounds': (1.5, 4, False),
 'peak_3_energy': 286.7,
 'peak_3_energy_bounds': (286.6, 286.9, True),
 'peak_3_width': 0.6,
 'peak_3_width_bounds': (0.3, 0.8, True),
 'peak_3_height': 0.5,
 'peak_3_height_bounds': (0.5, 1.3, True),
 'peak_4_energy': 287.5,
 'peak_4_energy_bounds': (287.4, 287.7, True),
 'peak_4_width': 1.5,
 'peak_4_width_bounds': (0.8, 1.5, True),
 'peak_4_height': 0.8,
 'peak_4_height_bounds': (0.2, 1, True),
 'peak_5_energy': 288.9,
 'peak_5_energy_bounds': (288.9, 289.1, True),
 'peak_5_width': 1.2,
 'peak_5_width_bounds': (0.8, 1.2, True),
 'peak_5_height': 1.65,
 'peak_5_height_bounds': (1.0, 2.9, False),
 'peak_6_energy': 290.4,
 'peak_6_energy_bounds': (290.2, 290.6, True),
 'peak_6_width': 0.8,
 'peak_6_width_bounds': (0.8, 1.2, True),
 'peak_6_height': 0.2,
 'peak_6_height_bounds': (0.1, 0.8, True),
 'peak_7_energy': 293.1,
 'peak_7_energy_bounds': (292.9, 293.3, True),
 'peak_7_width': 4.7,
 'peak_7_width_bounds': (3.2, 4.8, True),
 'peak_7_height': 0.6,
 'peak_7_height_bounds': (0.2, 0.8, True),
 'peak_8_energy': 296.9,
 'peak_8_energy_bounds': (295.9, 297, True),
 'peak_8_width': 4.8,
 'peak_8_width_bounds': (3.2, 4.8, True),
 'peak_8_height': 0.5,
 'peak_8_height_bounds': (0.2, 0.8, True),
 'peak_9_energy': 303,
 'peak_9_energy_bounds': (302.9, 303.1, False),
 'peak_9_width': 8.4,
 'peak_9_width_bounds': (6.4, 9.6, True),
 'peak_9_height': 0.5,
 'peak_9_height_bounds': (0.2, 0.8, True)},
    edge={'location': 289.5, 'height': 1.43, 'width': 0.6, 'decay': 0.01},
    baseline=0,
)

# notebook cell 33
SOG_CARBON_BAKE = dict(
    peaks={'peak_1_energy': 285.0,
 'peak_1_energy_bounds': (284.9, 285.1, False),
 'peak_1_width': 0.6,
 'peak_1_width_bounds': (0.4, 0.8, True),
 'peak_1_height': 0.2,
 'peak_1_height_bounds': (0.1, 0.3, True),
 'peak_2_energy': 285.5,
 'peak_2_energy_bounds': (285.4, 285.6, False),
 'peak_2_width': 0.6,
 'peak_2_width_bounds': (0.4, 1, True),
 'peak_2_height': 0.1,
 'peak_2_height_bounds': (0.05, 0.3, True),
 'peak_3_energy': 286.7,
 'peak_3_energy_bounds': (286.6, 286.9, True),
 'peak_3_width': 0.6,
 'peak_3_width_bounds': (0.3, 0.8, True),
 'peak_3_height': 0.2,
 'peak_3_height_bounds': (0.05, 1.3, True),
 'peak_4_energy': 287.7,
 'peak_4_energy_bounds': (287.5, 287.8, True),
 'peak_4_width': 1.5,
 'peak_4_width_bounds': (0.8, 1.6, True),
 'peak_4_height': 1.2,
 'peak_4_height_bounds': (0.2, 1.5, True),
 'peak_5_energy': 288.7,
 'peak_5_energy_bounds': (288.5, 289.1, True),
 'peak_5_width': 1.0,
 'peak_5_width_bounds': (0.8, 1.2, True),
 'peak_5_height': 1.4,
 'peak_5_height_bounds': (1.0, 2.9, True),
 'peak_6_energy': 292.1,
 'peak_6_energy_bounds': (291.5, 293.3, True),
 'peak_6_width': 2.8,
 'peak_6_width_bounds': (1.5, 4.8, True),
 'peak_6_height': 0.3,
 'peak_6_height_bounds': (0.1, 0.8, True),
 'peak_7_energy': 293.1,
 'peak_7_energy_bounds': (292.9, 293.3, True),
 'peak_7_width': 3.8,
 'peak_7_width_bounds': (2.8, 4.8, True),
 'peak_7_height': 0.15,
 'peak_7_height_bounds': (0.1, 0.8, True),
 'peak_8_energy': 296.9,
 'peak_8_energy_bounds': (295.9, 297, True),
 'peak_8_width': 4.8,
 'peak_8_width_bounds': (3.2, 4.9, True),
 'peak_8_height': 0.17,
 'peak_8_height_bounds': (0.05, 0.3, True),
 'peak_9_energy': 303,
 'peak_9_energy_bounds': (302.9, 303.1, False),
 'peak_9_width': 8.4,
 'peak_9_width_bounds': (6.4, 9.6, True),
 'peak_9_height': 0.18,
 'peak_9_height_bounds': (0.05, 0.3, True)},
    edge={'location': 289.5, 'height': 1.2, 'width': 0.6, 'decay': 0.01},
    baseline=0,
)

# notebook cell 36
SOG_SI_BAKE_DEV = dict(
    peaks={'peak_1_energy': 1848.3,
 'peak_1_energy_bounds': (1846, 1853, True),
 'peak_1_width': 0.8,
 'peak_1_width_bounds': (0.6, 1.2, True),
 'peak_1_height': 2.6,
 'peak_1_height_bounds': (2.2, 3, True),
 'peak_2_energy': 1849.4,
 'peak_2_energy_bounds': (1848.7, 1850.5, True),
 'peak_2_width': 1.6,
 'peak_2_width_bounds': (1.4, 1.8, True),
 'peak_2_height': 1.2,
 'peak_2_height_bounds': (1, 1.5, True),
 'peak_3_energy': 1865,
 'peak_3_energy_bounds': (1864, 1867, False),
 'peak_3_width': 7,
 'peak_3_width_bounds': (5, 9, True),
 'peak_3_height': 0.3,
 'peak_3_height_bounds': (0.1, 0.5, True)},
    edge={'location': 1847, 'height': 1.43, 'width': 0.6, 'decay': 0.025},
    baseline=0,
)

# notebook cell 39
SOG_SI_UV_DEV = dict(
    peaks={'peak_1_energy': 1848.3,
 'peak_1_energy_bounds': (1846, 1853, True),
 'peak_1_width': 1,
 'peak_1_width_bounds': (0.6, 1.2, False),
 'peak_1_height': 2.1,
 'peak_1_height_bounds': (1.7, 3, True),
 'peak_2_energy': 1849.4,
 'peak_2_energy_bounds': (1848.7, 1850.5, True),
 'peak_2_width': 1.6,
 'peak_2_width_bounds': (1.4, 1.8, False),
 'peak_2_height': 1.2,
 'peak_2_height_bounds': (1, 1.5, True),
 'peak_3_energy': 1865,
 'peak_3_energy_bounds': (1864, 1867, False),
 'peak_3_width': 7,
 'peak_3_width_bounds': (5, 9, True),
 'peak_3_height': 0.3,
 'peak_3_height_bounds': (0.1, 0.5, True)},
    edge={'location': 1847, 'height': 1.43, 'width': 0.6, 'decay': 0.025},
    baseline=0,
)

# notebook cell 42
SOG_SI_UV = dict(
    peaks={'peak_1_energy': 1843.3,
 'peak_1_energy_bounds': (1843, 1844, False),
 'peak_1_width': 1.8,
 'peak_1_width_bounds': (0.8, 1.8, True),
 'peak_1_height': 1.9,
 'peak_1_height_bounds': (1.0, 2.1, True),
 'peak_2_energy': 1844.5,
 'peak_2_energy_bounds': (1844, 1845, False),
 'peak_2_width': 0.9,
 'peak_2_width_bounds': (0.8, 1.8, True),
 'peak_2_height': 1.0,
 'peak_2_height_bounds': (1.0, 2.1, True),
 'peak_3_energy': 1845.6,
 'peak_3_energy_bounds': (1845, 1847, True),
 'peak_3_width': 1.6,
 'peak_3_width_bounds': (0.8, 3, True),
 'peak_3_height': 1.3,
 'peak_3_height_bounds': (0.7, 1.3, True),
 'peak_4_energy': 1846.6,
 'peak_4_energy_bounds': (1845, 1847, True),
 'peak_4_width': 1,
 'peak_4_width_bounds': (0.8, 3, True),
 'peak_4_height': 0.8,
 'peak_4_height_bounds': (0.2, 0.8, True),
 'peak_5_energy': 1848.2,
 'peak_5_energy_bounds': (1846, 1853, True),
 'peak_5_width': 1.0,
 'peak_5_width_bounds': (0.6, 1, False),
 'peak_5_height': 1.4,
 'peak_5_height_bounds': (0.8, 1.8, True),
 'peak_6_energy': 1849.2,
 'peak_6_energy_bounds': (1848.7, 1850.5, True),
 'peak_6_width': 1.6,
 'peak_6_width_bounds': (1.4, 1.8, False),
 'peak_6_height': 0.9,
 'peak_6_height_bounds': (0.3, 1.2, True),
 'peak_7_energy': 1853,
 'peak_7_energy_bounds': (1851, 1853.1, False),
 'peak_7_width': 2.1,
 'peak_7_width_bounds': (0.8, 2.1, True),
 'peak_7_height': 0.1,
 'peak_7_height_bounds': (0.1, 2.0, True),
 'peak_8_energy': 1854,
 'peak_8_energy_bounds': (1853, 1856, False),
 'peak_8_width': 4.2,
 'peak_8_width_bounds': (0.8, 6, True),
 'peak_8_height': 0.2,
 'peak_8_height_bounds': (0.1, 2.0, True),
 'peak_9_energy': 1865,
 'peak_9_energy_bounds': (1860, 1867, False),
 'peak_9_width': 9,
 'peak_9_width_bounds': (5, 9, True),
 'peak_9_height': 0.1,
 'peak_9_height_bounds': (0.05, 0.5, True)},
    edge={'location': 1847, 'height': 1.6, 'width': 0.6, 'decay': 0.02},
    baseline=0,
)

# notebook cell 45
SOG_SI_BAKE = dict(
    peaks={'peak_1_energy': 1843.3,
 'peak_1_energy_bounds': (1843, 1844, False),
 'peak_1_width': 1.8,
 'peak_1_width_bounds': (0.8, 1.8, True),
 'peak_1_height': 3.8,
 'peak_1_height_bounds': (3.0, 4.2, True),
 'peak_2_energy': 1844.5,
 'peak_2_energy_bounds': (1844, 1845, False),
 'peak_2_width': 0.9,
 'peak_2_width_bounds': (0.8, 1.8, True),
 'peak_2_height': 2.0,
 'peak_2_height_bounds': (1.7, 2.5, True),
 'peak_3_energy': 1845.6,
 'peak_3_energy_bounds': (1845, 1847, True),
 'peak_3_width': 1.6,
 'peak_3_width_bounds': (0.8, 3, True),
 'peak_3_height': 2.4,
 'peak_3_height_bounds': (1.7, 3, True),
 'peak_4_energy': 1846.6,
 'peak_4_energy_bounds': (1845, 1847, True),
 'peak_4_width': 1,
 'peak_4_width_bounds': (0.8, 3, True),
 'peak_4_height': 2,
 'peak_4_height_bounds': (1.7, 3, True),
 'peak_5_energy': 1848.2,
 'peak_5_energy_bounds': (1846, 1853, True),
 'peak_5_width': 1.0,
 'peak_5_width_bounds': (0.6, 1, False),
 'peak_5_height': 0.9,
 'peak_5_height_bounds': (0.3, 1.0, True),
 'peak_6_energy': 1849.2,
 'peak_6_energy_bounds': (1848.7, 1850.5, True),
 'peak_6_width': 1.6,
 'peak_6_width_bounds': (1.4, 1.8, False),
 'peak_6_height': 0.4,
 'peak_6_height_bounds': (0.1, 0.8, True),
 'peak_7_energy': 1853,
 'peak_7_energy_bounds': (1851, 1853.1, True),
 'peak_7_width': 2.1,
 'peak_7_width_bounds': (0.8, 2.1, True),
 'peak_7_height': 0.2,
 'peak_7_height_bounds': (0.05, 0.5, True),
 'peak_8_energy': 1854,
 'peak_8_energy_bounds': (1853, 1856, False),
 'peak_8_width': 4.2,
 'peak_8_width_bounds': (0.8, 6, True),
 'peak_8_height': 0.6,
 'peak_8_height_bounds': (0.1, 2.0, True),
 'peak_9_energy': 1860,
 'peak_9_energy_bounds': (1858, 1862, False),
 'peak_9_width': 4.2,
 'peak_9_width_bounds': (0.8, 6, True),
 'peak_9_height': 0.2,
 'peak_9_height_bounds': (0.1, 0.8, True),
 'peak_10_energy': 1865,
 'peak_10_energy_bounds': (1860, 1867, False),
 'peak_10_width': 9,
 'peak_10_width_bounds': (5, 9, True),
 'peak_10_height': 0.06,
 'peak_10_height_bounds': (0.05, 0.5, True),
 'peak_11_energy': 1871,
 'peak_11_energy_bounds': (1866, 1872, False),
 'peak_11_width': 9,
 'peak_11_width_bounds': (5, 9, True),
 'peak_11_height': 0.1,
 'peak_11_height_bounds': (0.05, 0.5, True)},
    edge={'location': 1847, 'height': 1.6, 'width': 0.6, 'decay': 0.018},
    baseline=0,
)


# ---------------------------------------------------------------- O K-edge (2026-11-02)
# No notebook model exists for the oxygen edge; seeds from the Brewer1/2 SOC LARIAT spectra (wide bounds).
# (energy, width, height, energy half-range, (width_min, width_max)); fit window 526-570 eV (>575 eV is a
# processing artefact of the post-edge normalisation).
OXYGEN_FIT_RANGE = (526.0, 570.0)
SOC_OXYGEN_SEEDS = [
    (531.3, 1.0, 0.3, 0.8, (0.4, 2.0)),   # pi* C=O
    (532.5, 1.2, 0.15, 0.8, (0.4, 2.0)),  # second low-energy feature (Bake plateau 531-532.5)
    (535.7, 1.5, 0.3, 1.0, (0.5, 3.0)),   # edge shoulder
    (537.6, 2.0, 0.4, 1.0, (0.8, 4.0)),
    (539.3, 2.5, 0.6, 1.0, (1.0, 5.0)),   # main sigma*
    (541.8, 3.0, 0.4, 1.5, (1.0, 6.0)),
    (546.0, 5.0, 0.3, 2.5, (2.0, 10.0)),
    (554.0, 8.0, 0.2, 5.0, (4.0, 16.0)),
    (569.0, 5.0, 0.04, 9.0, (2.0, 14.0)),  # post-edge bump seen at ~573 eV in both LARIAT spectra and as the isld upturn >558 eV (2026-11-02)
]
_OX_KEEP = {5: [0, 2, 3, 4, 5], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5, 6], 8: list(range(8)), 9: list(range(9))}


def soc_oxygen(n_peaks=8, merge56=False, wide_caps=False, split6=False):
    """Wide-bounds O K-edge model for SOC (5-9 Gaussians + erf edge with decay); all parameters free.

    wide_caps: p2 width cap 2 -> 3.5 eV and the 541.8 eV peak cap 6 -> 9 eV (they bound in the first co-fit).
    merge56: n_peaks=8 only; drop the 541.8 eV peak and let the 539.3 eV peak cover the whole main sigma*
    band (energy 538.5-541.5, FWHM 1-9 eV) -> 7 peaks, numbered p1-p4, p5 (merged), p6 (547), p7 (554.7).
    split6 (n_peaks=9, 2026-11-02): replace the 541.8 eV hump (fitted at 540.3 eV, FWHM 8.4 eV = unphysically wide) by two narrower peaks,
    541.2 eV (FWHM 1.5-6) and 545.0 eV (FWHM 2-7) -> 10 peaks p1-p5, p6 (541.2), p7 (545.0), p8 (547), p9 (554.6), p10 (572.9)."""
    keep = list(_OX_KEEP[n_peaks])
    if merge56:
        keep.remove(5)
    ent = []
    for i in keep:
        e, w, h, de, (w0, w1) = SOC_OXYGEN_SEEDS[i]
        elo, ehi = e - de, e + de
        if wide_caps and i == 1:
            w1 = 3.5
        if wide_caps and i == 5:
            w1 = 9.0
        if merge56 and i == 4:
            e, w, elo, ehi, w0, w1 = 539.6, 4.0, 538.5, 541.5, 1.0, 9.0
        if split6 and i == 5:
            if split6 == "wide":   # overlapping windows are safe only together with the energy-ordering constraint (nexafs_oxygen.OCoFit(ordered=True))
                ent.append((541.2, 3.5, 0.3, 539.5, 546.0, 1.5, 6.0))
                ent.append((545.0, 5.0, 0.2, 541.5, 549.0, 2.0, 7.0))
            else:
                ent.append((541.2, 3.5, 0.3, 539.7, 543.2, 1.5, 6.0))
                ent.append((545.0, 5.0, 0.2, 542.5, 547.5, 2.0, 7.0))
            continue
        if split6 == "wide" and i == 6:
            elo, ehi = 543.5, 551.0
        ent.append((e, w, h, elo, ehi, w0, w1))
    p = {}
    for j, (e, w, h, elo, ehi, w0, w1) in enumerate(ent, 1):
        p[f"peak_{j}_energy"] = e; p[f"peak_{j}_energy_bounds"] = (elo, ehi, True)
        p[f"peak_{j}_width"] = w; p[f"peak_{j}_width_bounds"] = (w0, w1, True)
        p[f"peak_{j}_height"] = h; p[f"peak_{j}_height_bounds"] = (0.0, 4.0, True)
    edge = dict(location=534.5, height=1.0, width=1.5, decay=0.0,
                location_bounds=(533.0, 536.5, True), height_bounds=(0.5, 1.6, True),
                width_bounds=(0.5, 4.0, True), decay_bounds=(0.0, 0.05, True))
    return dict(peaks=p, edge=edge, baseline=0)


# SOG (Si-O containing) O K-edge: overrides of soc_oxygen(9, wide_caps=True) (2026-11-02).  Unchanged SOC seeds on the LARIAT SOG spectra
# (OxygenSOG/lariat/cofit_soc_seeds*): p5/p6 pinned on the lower edges of their energy windows, p8 pinned at the 559 eV window ceiling and
# p9 at its 14 eV width cap (the SOG post-edge hump is at ~562 eV, FWHM ~15-20 eV; SOC has a 554 eV bump and a flat tail), RSS 0.218 (4 samples).
# index -> (seed E, seed W, seed H, E lo, E hi, W lo, W hi); peak numbering/count unchanged (9), windows kept non-overlapping.
SOG_OXYGEN_OVERRIDES = {
    4: (536.9, 1.5, 0.4, 536.0, 537.8, 0.8, 4.0),     # was 537.6 +-1 (536.6-538.6): the SOG resonance is ~1 eV lower
    5: (538.6, 2.5, 0.6, 537.8, 539.8, 1.0, 5.0),     # was 539.3 +-1
    6: (541.0, 3.0, 0.4, 539.8, 543.3, 1.0, 9.0),     # was 541.8 +-1.5, width cap 9 (wide_caps)
    8: (561.0, 10.0, 0.3, 549.0, 566.0, 4.0, 20.0),   # was 554 +-5, width 4-16; SOG hump ~562 eV
    9: (570.0, 8.0, 0.1, 562.0, 580.0, 2.0, 24.0),    # was 569 +-9, width 2-14; tail of the hump
}


def sog_oxygen(n_peaks=9, overrides=None):
    """O K-edge model for SOG: soc_oxygen(9, wide_caps=True) with the window/width overrides above (keys = 1-based peak number).
    The overrides replace seed and bounds of that peak only; the erf edge, p1-p3 and p7 are the SOC ones."""
    m = soc_oxygen(n_peaks, wide_caps=True)
    p = m["peaks"]
    for j, (e, w, h, elo, ehi, w0, w1) in (SOG_OXYGEN_OVERRIDES if overrides is None else overrides).items():
        p[f"peak_{j}_energy"] = e; p[f"peak_{j}_energy_bounds"] = (elo, ehi, True)
        p[f"peak_{j}_width"] = w; p[f"peak_{j}_width_bounds"] = (w0, w1, True)
        p[f"peak_{j}_height"] = h
    return m


def sog_oxygen7(decay=0.007, decay_free_cap=None):
    """7-peak O K-edge model for the SOG LARIAT spectra (2026-11-02; user choice after the p3/p4/p5 tests, OxygenSOG/scripts 06-10).
    Built from sog_oxygen(): edge-height floor 0.2, p8 width cap 30 (-> hump), then the 536.6/537.6/539.0 eV peaks reduced to two
    (p4 removed), then p3+p4 merged -> peaks: p1 531.4 (pi*), p2 532.5, p3 ~536.8 (narrow), p4 ~538.4, p5 ~541.5, p6 ~560 (hump), p7 ~572 (window-end tail).
    Edge decay is PINNED at `decay` /eV (user: the free fit ran to 0.04-0.05, edge -> 20 % by 576 eV, 'should decay gradually');
    decay_free_cap=X instead lets it float in (0, X)."""
    import copy
    ov = copy.deepcopy(SOG_OXYGEN_OVERRIDES)
    e, w, h, elo, ehi, w0, w1 = ov[8]; ov[8] = (e, w, h, elo, ehi, w0, 30.0)
    m = sog_oxygen(9, overrides=ov); p = m["peaks"]; ed = m["edge"]
    ed["height_bounds"] = (0.2, ed["height_bounds"][1], True)
    if decay_free_cap is None:
        ed["decay"] = decay; ed["decay_bounds"] = (decay - 5e-7, decay + 5e-7, True)
    else:
        ed["decay"] = min(decay, decay_free_cap / 2); ed["decay_bounds"] = (0.0, decay_free_cap, True)
    keys = ("energy", "energy_bounds", "width", "width_bounds", "height", "height_bounds")

    def drop(j, n):
        for i in range(j, n):
            for k in keys: p[f"peak_{i}_{k}"] = p[f"peak_{i + 1}_{k}"]
        for k in keys: del p[f"peak_{n}_{k}"]
    for j in (3, 5):                                               # kept peaks of the 536.6/539.0/541.0 group: wide window, width cap 6
        p[f"peak_{j}_energy_bounds"] = (534.7, 539.8, True); p[f"peak_{j}_width_bounds"] = (p[f"peak_{j}_width_bounds"][0], 6.0, True)
    p["peak_6_energy_bounds"] = (537.5, p["peak_6_energy_bounds"][1], True)
    drop(4, 9)                                                     # 8 peaks (best 8-peak model, 'drop4_p6free')
    p["peak_3_energy"], p["peak_3_width"], p["peak_3_height"] = 537.0, 2.0, 0.8       # merge slots 3+4 -> slot 3
    p["peak_3_energy_bounds"] = (535.0, 538.5, True); p["peak_3_width_bounds"] = (0.8, 6.0, True); p["peak_3_height_bounds"] = (0.0, 4.0, True)
    p["peak_5_energy"], p["peak_5_width"], p["peak_5_height"] = 539.5, 4.0, 0.6
    p["peak_5_energy_bounds"] = (537.5, 541.5, True); p["peak_5_width_bounds"] = (1.0, 7.0, True)
    p["peak_6_energy_bounds"] = (541.5, 548.5, True)
    drop(4, 8)                                                     # 7 peaks
    return m


def sog_oxygen8(decay=0.007):
    """8-peak O K-edge model for the LARIAT SOG spectra (2026-11-02): sog_oxygen7 + an extra peak in 548-558 eV (fitted ~556.3 eV, FWHM ~5.8) that removes the
    +-0.015 ripple at 545-565 eV of the 7-peak fit (RSS 0.129 -> 0.088); p6 (hump) window narrowed to 555-566.  Edge decay pinned (`decay`).
    LARIAT-ONLY model (user, 2026-11-02): the RSoXR isld models are built separately (OxygenSOG/scripts/14), they do not inherit this peak's energy/width."""
    m = sog_oxygen7(decay=decay); p = m["peaks"]
    p["peak_6_energy_bounds"] = (555.0, 566.0, True)
    p["peak_8_energy"], p["peak_8_width"], p["peak_8_height"] = 553.0, 4.0, 0.15
    p["peak_8_energy_bounds"] = (548.0, 558.0, True); p["peak_8_width_bounds"] = (1.5, 10.0, True); p["peak_8_height_bounds"] = (0.0, 4.0, True)
    return m
