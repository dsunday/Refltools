# RSoXR reduction criteria: a walk-through

This guide explains, step by step, how `rsoxr_reduce` turns raw ALS files into
stitched reflectivity curves. It covers every decision, the threshold behind it,
the flag it raises and the reason for it. It goes with the
[`rsoxr-reduce` skill](SKILL.md): the skill says *what to run*, this guide says
*what the tool decides and why*.

Every threshold below comes from the code (as of Oct 2026). Defaults were tuned on a
reference dataset, the Nov 2025 ALS beamtime; numbers quoted from it are labelled
"Nov 2025". Each one is labelled in one of two ways:

- **`setting.key`**: stored in the project's `reduction.json` and changeable with
  `python -m rsoxr_reduce set <key> <value> -p <project>`. Defaults are in
  `rsoxr_reduce/settings.py`.
- *hard-coded*: in the source and not changeable with `set`.

---

## 0. The big picture

```
raw folder ──► classify files ──► pick one scan per energy × section
                                          │
            ┌─────────────────────────────┘
            ▼
  trim each section ──► subtract PD background ──► ÷ open beam (I0)
            │
            ▼
  stitch sections (median ratio in each overlap) ──► Q = 4π sin θ / λ
            │
            ▼
  quality checks ──► ok: write <sample>_<E>eV.dat
                     review: hold until a person looks
                     rejected: never written
```

**Flag severity**

| Severity | Meaning |
|---|---|
| **info** | Recorded in `reduction_record.csv` and the report. The energy is still written. |
| **review** | The energy is held and no `.dat` is written until someone looks at the diagnostic and runs `accept` or `override`. |
| **reject** | The energy can't be reduced (e.g. no photodiode scan). |

**Detectors.** Each energy is measured in **sections**: a photodiode (**PD**) scan
at low angle, where the beam is bright, then one or more channeltron (**CEM**,
called **CT1, CT2, ...**) scans at higher angles. Adjacent sections overlap in
angle, and that overlap is what lets them be stitched together.

---

## 1. Classifying raw files (`inventory.py`, `classify`)

Every file is classified from the beamline log and its data, never from its name.
The one exception is that an energy scan with "open" and "beam" in its name is an
open beam.

| Class | Rule | Setting |
|---|---|---|
| `UNLOGGED` | file not in the beamline log | none |
| `EMPTY` | < **200 bytes**, or 0 points | `classify.min_bytes` |
| `ZERO` | no positive intensity at all | none |
| `ENERGY_SCAN` | column 0 max > **100** (i.e. eV) and span > **5 eV** | `classify.energy_min`, `classify.energy_scan_min_span` |
| `OPEN_BEAM` | an energy scan named "open…beam", or taken at θ ≈ 2θ ≈ 0 (\|θ\|, \|2θ\| < 0.05°, *hard-coded*) | none |
| `DUMMY` | ≤ **5** points **or** max angle ≤ **1°** (the `d 0 -> 1 n=2` filler lines in scripts) | `classify.dummy_max_points`, `classify.dummy_max_angle` |
| angle scan | max angle ≤ **90°**, min angle ≥ −1° (*hard-coded*), script scan type | `classify.angle_max` |
| `ALIGNMENT` | anything else (motor scans, etc.) | none |

Only angle scans go on to reduction. Energy scans go to `escan`, and open beams
are used for normalisation.

---

## 2. Choosing one scan per energy and section (`selection.py`)

**Grouping by energy.** Files within **0.05 eV** of each other belong to the same
energy (`group.energy_tol`). Set-point jitter is < 0.004 eV and energy grids are
≥ 0.2 eV apart, so this is safe.

**Finding the sections.** For each detector, scans are clustered by how much their
angular ranges overlap: intersection over union ≥ **0.6** (*hard-coded*). Count
time is ignored, because it changes with energy. A cluster that appears at
≥ **25 %** of the energies is a real ("nominal") section (`group.section_min_support`).
Sections are ordered by start angle, PD first, which is also the stitch order.

**Choosing a file.** For each energy × section:

| Code | Rule | Severity |
|---|---|---|
| `SUPERSEDED` | Another full-range scan of the same section was taken **later**. The latest wins (group convention: a re-measurement was re-measured for a reason). | file not used |
| `PARTIAL_RANGE` | Span < **80 %** of the section's nominal span (`select.partial_frac`). Used only if no full-range scan exists. | **review** if it ends > 0.6° early (`group.section_match_deg`) or leaves a gap to the section below, otherwise info |
| `OFF_SECTION` | Angular range matches no nominal section | file not used |
| `NO_PD` | No photodiode section, so the energy can't be normalised | **reject** |
| `MISSING_SECTION` | A *middle* section is absent, leaving a gap in Q | **review** |
| `TRUNCATED_Q` | A *trailing* section is absent, so Q simply ends early | info |
| `MIXED_SESSION` | The chosen scans were taken > **6 h** apart (`select.off_date_hours`) | **review** |
| `POSITION_MISMATCH` | Sections measured > **0.2 mm** apart on the sample (`group.position_tol`). The spot is often stepped on purpose to limit beam damage, so this is only held above **3 mm** (`group.position_review_mm`). | info, or **review** > 3 mm |

---

## 3. Where the PD data start: beam clipping (`trim.py`, `pd_start`)

**The physics.** At very low angle the beam footprint is longer than the sample,
so part of the beam misses it. The PD signal *rises* with angle until the
footprint fits, peaks, and then follows R(θ). Data before the peak are not
reflectivity.

**The rule.** Clipping depends on geometry, not energy, so **one clip angle is
used for the whole folder**:

1. For each energy, find the angle of the PD maximum below **1.5°** (carbon) or
   **1.2°** (oxygen) (`trim.clip_search_deg`). At least 3 points are needed in that
   range (*hard-coded*).
2. The folder clip angle is the **median** of those peak angles.
3. Each PD scan starts at its first point at or above the clip angle (within half
   a grid step). The θ ≈ 0 direct-beam point is never kept (`trim.min_start` = 1).

On Nov 2025 data this matches the manual start within one grid step in 82 % of
carbon folders. Manual oxygen choices scatter by ±0.17°.

| Flag | Trigger | Severity |
|---|---|---|
| `CLIP_SHIFT` | This energy's own peak is > **2 grid steps** from the folder angle (`trim.clip_shift_steps`) | info |
| `PD_SATURATION` | PD ≥ **9.99 V** (`trim.pd_sat_level`; the ADC tops out at 10.0005 V). Start after the saturated plateau. | **review** if the plateau extends past the clip angle, because real data are lost and the PD gain should be lowered next time. Otherwise info. |
| `NO_CLIP_PEAK` | No PD maximum found below the search angle, so only the θ ≈ 0 point is dropped | **review** |

Change the clip angle for the whole folder with `folder --clip-angle <deg>`.

---

## 4. PD background: the dark offset (`trim.py`, `fit_pd_background`)

**The physics.** The photodiode reads a small constant voltage even with no beam.
That offset has to be subtracted before dividing by I0. If it is wrong, the PD's
weak high-angle end is badly distorted, and so is the scale factor onto CT1,
because that overlap sits right where the PD signal is weakest.

**The method.** Where PD and CT1 overlap, both see the same reflected beam, so
`PD = background + k × CEM rate`. A straight-line fit of PD against the CEM rate
gives the background as its **intercept**. The scatter of the residuals gives the
PD noise σ, which the noise floors in §5 use.

Per energy:

| Step | Threshold |
|---|---|
| Points used | PD points inside CT1's angular range; need ≥ **6** (*hard-coded*) |
| CEM points must be well measured | counts ≥ **200** (`trim.bg_fit_min_counts`) |
| ... and in the linear range | rate ≤ **1×10⁵ cps** (`trim.ct_max_rate`) |
| Robust fit | 6 passes, dropping points > **3σ** from the line (σ = 1.4826 × MAD). Fails if < 5 points remain or the slope k ≤ 0 (*hard-coded*). |

Per folder (`folder_background`):

| Step | Threshold |
|---|---|
| A fit counts as "well-conditioned" if... | the signal range it spans is > **5σ** (*hard-coded*) **and** σ < **0.003 V** (`trim.bg_fit_max_sigma`) |
| Folder background | **median** over the well-conditioned energies; needs ≥ **3** of them (`trim.bg_fit_min_energies`) |
| Otherwise, fallback | C **0.0024 V**, O **0.0020 V**, other **0.0022 V** (`trim.bg_default`, the Nov 2025 medians), with PD σ = **0.0008 V** (`trim.pd_sigma_default`; the ADC step is 3.05×10⁻⁴ V) |

The report records which source was used ("pd-vs-ct regression" or "default").

**Nov 2025 values (reference data):** dark offsets were about 0.0022–0.0028 V, but
about 0.006 V in one carbon session (samples Brewer9/10/13/15). Check your own
session's value in the `reduce` report. Manual reductions often used a lower background,
which over-scales CT at absorbing energies.

Override for the whole folder with `folder --pd-background <V>`, or for one energy
with `override <E> --bg <V>`.

---

## 5. Noise floors: when data are "too noisy to keep" (`trim.py`)

Every noise test uses a **smoothed** signal, so that a single noisy point can't
trigger a cut: a running median over 5 points, then a running mean over 3
(*hard-coded*).

| Detector | Per-point error model | Usable while... | Equivalent per-point error |
|---|---|---|---|
| PD | σ / (smoothed PD − background) | SNR ≥ **5** (`trim.snr_min`) | ≈ 20 % |
| CEM | 1 / √(smoothed counts), counts = rate × count time | counts ≥ **10** (`trim.ct_min_counts`) | ≈ 32 % |

These floors set the **noise cap** for each section: the last point that is still
usable. They are used in two places:

- **At a junction** (§7), the lower section never runs past its noise cap, except
  as needed to get enough overlap points.
- **At the tail** of the last section, everything after the last usable point is
  cut. The final point of the scan is **always** dropped (`trim.ct_end_min` = 1,
  the legacy convention).

| Flag | Trigger | Severity |
|---|---|---|
| `LOW_COUNTS_TAIL` | > **20 %** of the last section was trimmed as below the floor (`qc.low_counts_tail_frac`) | info |

To cut noisy tails more aggressively, raise the floors, e.g. `set trim.ct_min_counts 25`
(about 20 % per point) or `set trim.snr_min 10`.

---

## 6. CEM saturation (`trim.py`, `ct_start`)

The channeltron becomes non-linear at high count rates. In Nov 2025 the PD/CEM
ratio was off by **+3 % at 1–3×10⁵ cps** and **+22 % at 3×10⁵–1×10⁶ cps**.

**Rule:** leading CEM points with rate > **1×10⁵ cps** (`trim.ct_max_rate`) are trimmed.

| Flag | Trigger | Severity |
|---|---|---|
| `CT_SATURATION` | leading points above the limit were trimmed | info, or **review** if later points (after the start) also exceed it |

The same limit keeps saturated points out of the background fit (§4).

---

## 7. Stitching junctions

A **junction** is where one section hands off to the next: PD→CT1, CT1→CT2, and
so on. At each junction three things happen, in order.

### Step 1: where to end the lower section (`trim.py`, `junction_end`)

Higher in angle the lower section gets noisier, while the upper section counts
better. The aim is to keep *just enough* of the lower section in the overlap to
pin down the scale factor, then hand over to the upper section.

1. Find the lower section's **noise cap** (§5).
2. Walk forward through the overlap one lower-section point at a time. At each
   candidate end angle, count the upper-section points between the start of the
   overlap and that angle (**n**), and predict how well the scale would be known:

   σ_scale ≈ 1.25 × rms(combined per-point relative error) / √n

3. **Stop at the first point where n ≥ 4** (`trim.min_overlap`) **and σ_scale ≤ 2 %**
   (`trim.scale_rel_target`).
4. If the noise cap comes first:
   - keep the lower section up to the cap;
   - if that gives fewer than 4 upper points, extend past the cap until there
     are 4 → `LOW_OVERLAP` (info);
   - if the predicted σ_scale is still > **4 %** (2 × target) → `NOISY_OVERLAP` (info).
5. The lower section's own final point is never kept.
6. If the upper section starts beyond the end of the lower one → `NO_OVERLAP`
   (**review**; see Step 3).

### Step 2: the scale factor (`stitch.py`)

- At each overlapping angle, take the ratio between the already-stitched lower
  curve and the upper section.
- **Scale = median of those ratios.** Stitching is cumulative, so CT2 is scaled
  onto PD+CT1, not onto CT1 alone.
- **Measured uncertainty:** σ = 1.25 × 1.48 × MAD / median / √n. This comes from
  the actual scatter of the ratios, whereas Step 1 only predicted σ from the
  noise model.

This is numerically identical to the old widget: trim → background → ÷ I0 →
drop ≤ 0 → cumulative median-ratio stitch → Q with λ = 12399/E.

### Step 3: checking the junction (`qc.py`, `energy_flags`; `pipeline.py`)

| Condition | Flag | Severity | What happens |
|---|---|---|---|
| no overlap ratios (n = 0) | `NO_OVERLAP` | **review** | try the gap bridge, then the trend (§8) |
| fewer than 4 ratios | `LOW_OVERLAP` | info | scale used as is |
| measured σ > **10 %** (`qc.noisy_overlap`) | `NOISY_OVERLAP` | info | scale blended with the neighbour-energy trend, weighted by 1/σ² → `SCALE_BLENDED` (info; `qc.blend_noisy_scales`) |
| ≥ 6 ratios (*hard-coded*), and the median ratio of the second half of the overlap differs from the first half by > **20 %** (`qc.jump`) **and** by > **3×** its own noise (*hard-coded*) | `JUMP` | **review** | The two detectors disagree on the *shape* of the curve (CEM non-linearity, misalignment), which no single scale can fix |
| points ≤ 0 after background subtraction | `NEG_AFTER_BG` | info | those points are dropped |

**Why two noisy-overlap thresholds?** The 4 % in Step 1 is a *prediction* from the
noise model and only describes how the trim went. The 10 % in Step 3 is the
*measured* ratio scatter, and it is the one that triggers blending. Both use the
same `NOISY_OVERLAP` code.

### Worked example: PD→CT1 at 285 eV

1. CT1 starts at 8°. PD stays above SNR 5 until 10.5° (its noise cap).
2. Walking forward from 8°: at 8.6° there are 4 CT1 points but the predicted σ is
   3 %; at 8.9° there are 6 points and σ is 1.9 %. **PD ends at 8.9°.**
3. The median of the 6 PD/CT ratios gives the CT1 scale. Measured σ = 2.5 %, which
   is < 10 %, so there is no blending.
4. Six ratios is enough for the drift test: the first three average 1.00 and the
   last three 1.03, a 3 % drift, which is below 20 %. **No `JUMP`.**
5. The scale lies within 30 % of the 280–291 eV trend (§9), so **the energy passes.**

---

## 8. When sections don't overlap (`qc.py`, `pipeline.py`)

When `NO_OVERLAP` occurs and `qc.use_trend_for_no_overlap` is on (the default), the
pipeline tries two fallbacks in order. **Both stay held for review.**

1. **Bridge the gap: `SCALE_FROM_GAP`.** Used if the gap is ≤ **0.5°**
   (`qc.max_gap_deg`). Fit ln R against θ with a straight line through the last
   **4** points of the lower curve and the first 4 of the upper curve
   (`qc.gap_points`), extrapolate both to the middle of the gap, and match them
   there.
2. **Borrow from neighbours: `SCALE_FROM_TREND`.** Take the median gain-normalised
   scale of the **3** nearest energies whose junction did overlap (*hard-coded*),
   corrected for this energy's PD gain.

---

## 9. Scale outliers across energies (`qc.py`, `trend_flags`)

The PD/CEM sensitivity ratio varies **smoothly** with energy, so the scale factor
at one junction (e.g. PD→CT1) should follow a smooth trend across the energy
series. Scales are compared after normalising out the PD gain, which is in decades.

**The trend at energy E:**

- Neighbours: up to **3** energies on each side (`qc.trend_window`), and only those
  within **6 eV** (`qc.trend_max_de`). E itself is excluded.
- Needs ≥ 3 neighbours (*hard-coded*).
- A robust line through them (Theil–Sen slope, median intercept), done in log.
  The slope is used only when there are neighbours on *both* sides; at the end of
  the energy range the trend is flat, never extrapolated.
- Trend scatter σ_trend = 1.4826 × MAD of the residuals, with a floor of **0.05**
  (about 5 %; `qc.trend_sigma_min`).

**`SCALE_OUTLIER` (review)** fires when

  |ln(scale / trend)| > max( **3** × √(σ_own² + σ_trend²), ln **1.30** )

(`qc.scale_outlier_mad`, `qc.scale_outlier_min`). In words, a scale is an outlier
only when it is *both* more than 3σ off *and* more than 30 % off. The 30 % floor is
deliberate: detector sensitivity has real structure at the carbon edge, and a
tighter limit would hold genuine features. Because the test is in log, the limit
is really **+30 % / −23 %**.

---

## 10. Filter changes and fringe damping (`qc.py`, `filter_checks`)

**The concern.** When the filter or aperture changes between two sections, the
higher-flux section can be over-exposed or see a larger footprint. Its Kiessig
fringes then look **damped**, and the stitch is biased. This is the group's
experience at the beamline, so a filter change is **always held for review**.

| Flag | Trigger | Severity |
|---|---|---|
| `FILTER_CHANGE` | filter/aperture differs between adjacent sections | **review** |

**The fringe test** (when both sections have ≥ **8** positive points in the
overlap, *hard-coded*):

1. Put both curves on the coarser section's angle grid, take ln R, remove a cubic
   trend, and smooth over 3 points. What is left is the fringes.
2. **Fringe contrast** = rms of those residuals, in ln R. Below **0.03** (about 3 %;
   `qc.fringe_min_contrast`) there is nothing to compare.
3. **Correlation** between the two fringe patterns, and the **amplitude ratio**
   upper/lower (a regression through the origin).

| Flag | Trigger | Severity |
|---|---|---|
| `FRINGE_CHECK_SKIPPED` | both contrasts < 0.03, or too few points even for the fallback | info |
| `FRINGE_DAMPING` | correlation ≥ **0.5** (`qc.fringe_min_corr`) **and** ratio < **0.75** (`qc.fringe_damping` = 0.25 → 1 − 0.25) | **review** |
| `FRINGE_MISMATCH` | correlation < 0.5: the fringes don't line up, or are too noisy to compare | **review** |
| `FRINGES_MATCH` | correlated and not damped: the change looks harmless | info |

**Fallback when the overlap is too short:** compare the fringe rms in **6°**
windows on each side of the junction (`qc.fringe_window_deg`; quadratic detrend).
This test is weaker, because R itself changes between the two windows:
same-filter junctions scatter 0.4–1.8 on it.

| Flag | Trigger | Severity |
|---|---|---|
| `FRINGE_DAMPING` | lower-side rms ≥ 0.03 and ratio < 0.75 | **review** |
| `FRINGE_CHECK_WEAK` | anything else | info |

**What to do with a `FILTER_CHANGE`:** look at the overlap panel of the diagnostic.
If the fringes are damped, either trim or drop the damped section's overlap, or
accept with a note. The user decides.

---

## 11. Open beam and gain (`pipeline.py`, `choose_open_beam`)

Reflectivity = sample signal ÷ open-beam signal (I0) at the same energy. The open
beam is chosen by ranking candidates on:

1. **same edge** (required),
2. same filter,
3. same exit-slit gap, within **±3 %** (`open_beam.slit_tol`),
4. covering this energy,
5. closest in time to the PD scan.

**PD gain is in decades.** If the sample and open beam were taken at different
PD gains, I0 is multiplied by **10^(gain_sample − gain_open-beam)**. That is an
exact correction (`open_beam.gain_correct`).

| Flag | Trigger | Severity |
|---|---|---|
| `GAIN_CORRECTED` | gains differed and were corrected | info |
| `GAIN_MISMATCH` | gains differed and correction is turned off | **review** |
| `SLIT_MISMATCH` | best open beam's slit gap is outside ±3 %; the absolute scale is off and can't be corrected | **review** |
| `FILTER_MISMATCH` | best open beam has a different filter | **review** |
| `OB_EXTRAP` | the energy is outside the open beam's energy range | **review** |
| `NO_OPEN_BEAM` | no open beam for this edge; the curve is normalised to its maximum | **review** |

**Errors** written to the `.dat` are a flat **1 %** of R (`output.error_frac`), the
same as the old widget.

**Nov 2025 note (reference data):** the manual reductions used gain-8 open beams for gain-9 sample
data, so their R is about 10× too high (×1.2–1.3 more where the slit also differed).
ReducedAuto files are gain-corrected, with R ≈ 0.7–1 at low q.

---

## 12. Quick reference: every flag

| Flag | Severity | Trigger (short) | Section |
|---|---|---|---|
| `DUMMY` / `EMPTY` / `ZERO` / `UNLOGGED` / `ALIGNMENT` | file not used | not a usable angle scan | §1 |
| `ENERGY_SCAN` / `OPEN_BEAM` | routed | energy scan → `escan` / normalisation | §1 |
| `SUPERSEDED` | file not used | a later full-range scan exists | §2 |
| `OFF_SECTION` | file not used | range matches no section | §2 |
| `PARTIAL_RANGE` | info / review | span < 80 %; review if it ends early or leaves a gap | §2 |
| `NO_PD` | reject | no photodiode section | §2 |
| `MISSING_SECTION` | review | middle section absent | §2 |
| `TRUNCATED_Q` | info | trailing section absent | §2 |
| `MIXED_SESSION` | review | scans > 6 h apart | §2 |
| `POSITION_MISMATCH` | info / review | > 0.2 mm apart; review > 3 mm | §2 |
| `CLIP_SHIFT` | info | own PD peak > 2 steps from the folder clip angle | §3 |
| `PD_SATURATION` | info / review | PD ≥ 9.99 V; review if data past the clip angle are lost | §3 |
| `NO_CLIP_PEAK` | review | no low-angle PD maximum | §3 |
| `LOW_COUNTS_TAIL` | info | > 20 % of the last section trimmed | §5 |
| `CT_SATURATION` | info / review | CEM > 1e5 cps at the start; review if also later | §6 |
| `NO_OVERLAP` | review | sections don't overlap | §7–8 |
| `LOW_OVERLAP` | info | < 4 overlap points | §7 |
| `NOISY_OVERLAP` | info | scale σ > 4 % (predicted) or > 10 % (measured) | §7 |
| `SCALE_BLENDED` | info | noisy scale blended with the trend | §7 |
| `JUMP` | review | ratio drifts > 20 % and > 3σ across the overlap | §7 |
| `NEG_AFTER_BG` | info | points ≤ 0 after background, dropped | §7 |
| `SCALE_FROM_GAP` | review | scale from bridging a ≤ 0.5° gap | §8 |
| `SCALE_FROM_TREND` | review | scale from the 3 nearest energies | §8 |
| `SCALE_OUTLIER` | review | > 3σ and > 30 % off the energy trend | §9 |
| `FILTER_CHANGE` | review | filter/aperture differs between sections | §10 |
| `FRINGE_DAMPING` | review | correlated fringes, amplitude ratio < 0.75 | §10 |
| `FRINGE_MISMATCH` | review | fringe correlation < 0.5 | §10 |
| `FRINGES_MATCH` | info | fringes agree | §10 |
| `FRINGE_CHECK_WEAK` / `FRINGE_CHECK_SKIPPED` | info | short overlap / no measurable fringes | §10 |
| `GAIN_CORRECTED` | info | PD gain difference corrected | §11 |
| `GAIN_MISMATCH` / `SLIT_MISMATCH` / `FILTER_MISMATCH` / `OB_EXTRAP` / `NO_OPEN_BEAM` | review | open-beam problems | §11 |

---

## 13. Changing a threshold

```bash
python -m rsoxr_reduce set <key> <value> -p <project>
python -m rsoxr_reduce reduce -p <project>
```

| If you want to... | Change | Effect |
|---|---|---|
| cut noisy tails harder | raise `trim.ct_min_counts` (e.g. 25) or `trim.snr_min` (e.g. 10) | shorter curves; more `LOW_COUNTS_TAIL` |
| demand a better-determined stitch | lower `trim.scale_rel_target` (e.g. 0.01) or raise `trim.min_overlap` | longer overlaps; more `LOW_OVERLAP` / `NOISY_OVERLAP` |
| catch smaller scale errors | lower `qc.scale_outlier_min` (e.g. 0.15) | more `SCALE_OUTLIER` holds, including real edge structure |
| be stricter about fringe damping | lower `qc.fringe_damping` (e.g. 0.15 → ratio < 0.85) | more `FRINGE_DAMPING` holds |
| catch smaller shape mismatches | lower `qc.jump` (e.g. 0.10) | more `JUMP` holds |
| trust the CEM to higher rates | raise `trim.ct_max_rate` | fewer leading points trimmed; also affects the background fit |
| allow a slightly different slit for the open beam | raise `open_beam.slit_tol` | fewer `SLIT_MISMATCH` |

Folder-wide values (clip angle, PD background) have their own command:
`folder --clip-angle` / `folder --pd-background`. Single-energy fixes go through
`override` with a `--reason`.

---

## 14. Known issues (documented, not yet fixed)

1. **Stale docstring.** `qc.py` says the `SCALE_OUTLIER` floor is 15 %; the code
   and settings use 30 %.
2. **One code, two meanings.** `NOISY_OVERLAP` fires at 4 % *predicted* σ (trim)
   and at 10 % *measured* σ (QC). The record doesn't say which.
3. **Loose noise floors compared with the written errors.** A floor of 10 counts is
   about 32 % per point and SNR 5 about 20 %, and both are tested on smoothed data.
   The `.dat` still carries a flat 1 % error, so tail points are over-weighted in
   fitting.
4. **`NEG_AFTER_BG` is info only.** Dropping points ≤ 0 on the PD floor biases R
   upward there. It happens most when the background is set too high.
5. **The background spread is never checked.** `folder_background` computes the
   spread of the per-energy backgrounds, but no flag uses it. If the fit falls back
   to the default, a session with a high offset (≈ 0.006 V in one Nov 2025 carbon session) would
   get 0.0024 V.
6. **`FRINGE_MISMATCH` can't tell noise from a real mismatch.** A noisy upper
   section lowers the correlation and causes a review hold even when the fringes
   are fine.
7. **Hard-coded cutoffs.** These can't be changed with `set`: the ≥ 6 / ≥ 8 point
   minimums, 3σ clipping, the smoothing windows, IoU ≥ 0.6, k = 3 nearest energies,
   and the 3× drift-noise factor in `JUMP`.
8. **Cosmetic.** In `settings.py` the comment for `qc.fringe_min_corr` sits on the
   `fringe_window_deg` line.
