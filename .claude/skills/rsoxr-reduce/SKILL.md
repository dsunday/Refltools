---
name: rsoxr-reduce
description: Reduce raw soft X-ray reflectivity (RSoXR) beamline data with Refltools from plain-text requests — classify raw files (angle scans, energy scans, open beam, dummy/failed runs), pick one scan per energy and detector section, trim beam clipping / noise floors / CEM saturation by explicit rules, stitch photodiode + channeltron sections, normalise to the open beam, flag problem energies (filter changes, fringe damping, bad overlaps, scale outliers) and reduce energy scans. Use whenever the user talks about data reduction, stitching, trimming, raw ALS data, photodiode/channeltron/CEM sections, open beam, Reduced/ReducedAuto folders, or reducing a sample/edge (e.g. "reduce <sample> <edge>", "reduce sample S1 oxygen", "why was <E> eV trimmed there").
---

# RSoXR reduction harness (Refltools/rsoxr_reduce)

You translate plain-text requests into calls on `rsoxr_reduce`, a project-folder
front end that replaces the manual `RSoXRProcessor` + `RSoXRTrimWidget` workflow.
Every decision (file used / not used, each trim, background, scale, open beam)
carries a reason string, and problem energies are held back until reviewed.
**Do not hand-tune trims in notebook code when a harness call exists**; use
`override` so the reason is recorded. Module docstrings in
`$REFLTOOLS/rsoxr_reduce/*.py` document every rule.

**Human-readable walk-through of every criterion** (thresholds, settings keys, flags,
why): [`REDUCTION_GUIDE.md`](REDUCTION_GUIDE.md) in this folder. Read it when the user
asks to explain a trim/flag/threshold or wants to change a setting; quote it to them.

## Environment

**Setup (per user):** `PY` = the python of your refnx environment (the one with the
local refnx fork installed), `REFLTOOLS` = your Refltools checkout. If the user hasn't
said, look for them (e.g. `CLAUDE.md`, an active conda env) or ask.

```bash
PY=<your env>/bin/python          # e.g. /homes/<user>/mambaforge/envs/refnxlocal/bin/python
REFLTOOLS=<your Refltools checkout>   # e.g. /homes/<user>/Refltools
cd $REFLTOOLS
$PY -m rsoxr_reduce <command> -p <project> ...
```
Python API: `from rsoxr_reduce.project import ReductionProject`.

## The project

One project per **raw folder** = one sample × edge × round. A typical layout is
`<beamtime>/<sample>/<edge>[N]`, e.g. `.../<beamtime>/<sample>/Carbon`, `.../Carbon2`
(a second carbon round), `.../Oxygen`. The project folder
is the output folder, default `<raw>/ReducedAuto/`:
`reduction.json` (sources, settings, overrides, last-run summary),
`<sample>_<E>eV.dat` (same format as the widget output, loadable by
`rsoxr_harness init`), `reduction_record.csv` (one row per energy × section:
file, trims, angles, reasons, background, scale, flags, status),
`reduction_report.md`, `angular_ranges.csv` (legacy layout),
`diagnostics/<E>eV.png`, `overview_scales.png`, `overview_waterfall.png`,
`EnergyScans/`. **Never write into an existing `Reduced/` folder** (the tool refuses).
The beamline log is found by walking up from the raw folder; open beams from an
`OpenBeam*`/`Open Beam` folder nearby, else the log's folder (`--open-beam` overrides).

> **Current limitation: raw data must be pre-sorted by hand.** The harness assumes the
> raw files are already sorted into one folder per sample × edge × round, because that
> is how the data were organised when it was built. It can't yet take an unsorted
> beamtime dump and split it.
>
> **Important feature to implement later: automatic sorting from the beamline log.**
> The log already records, for every file, what's needed to sort it (energy → edge,
> sample position x/y, scan type, detector, timestamp). The planned feature is a
> command that reads the log, groups files into sample × edge × round, proposes the
> grouping with reasons for the user to confirm, then creates the projects. Build it on
> the existing parsing and classification in `inventory.py` (`parse_log`, `file_info`,
> `edge_of`).
>
> **Until then:** if a user points at an unsorted folder, don't sort or move files
> yourself. Say this isn't supported yet and ask them to pre-sort the data.

## Intent → action

| User says | Do |
|---|---|
| "reduce <sample> <edge>" (e.g. "reduce S1 oxygen") / "reduce <folder>" | `init "<raw folder>" [--sample NAME]` (if no project yet), then `reduce -p <raw>/ReducedAuto`. Report status counts, the held/rejected energies with their flag codes, folder-wide clip angle and PD background (+ source), and link `overview_scales.png`, `overview_waterfall.png` and the report. `Read` the overview PNGs first. |
| "what's in this folder" / "which runs failed" | `inventory` (non-angle files with reasons; `--all` for every file), `select` (chosen file per energy/section + every file not used, with its code). |
| "why was <E> eV trimmed there" / "explain E" | `why <E> -p P` — trims, reasons, files not used, open-beam choice, junction scales, flags. Show the diagnostic PNG (`reduce --energies <E> --diagnostics all` if it doesn't exist yet). |
| "show me the flagged energies" | `review -p P`; `Read` each diagnostic PNG and summarise what you see per energy (1–2 lines). |
| "start PD at 6 at <E>" / "cut CT1 at -20" / "background 0.004" | `override <E> --section PD --start 6 --reason "..."` (legacy index convention: `end=-1` drops the last point); `--bg`, `--scale`; then `reduce --energies <E> --diagnostics all` and show the PNG. Always pass the user's reason. |
| "use the other PD scan at <E>" | `select --energy <E>` to list candidates, then `override <E> --section PD --use <file> --reason "..."`. |
| "drop CT2 at <E>" | `override <E> --drop CT2 --reason "..."`. |
| "use the open beam from <day/run> for <E>" | `override <E> --open-beam <name> --reason "..."` (gain difference is still corrected; slit/filter mismatch is flagged). |
| "<E> looks fine, accept it" | only after you (and the user) looked at its diagnostic: `accept <E> --reason "..."`, then `reduce`. |
| "set the clip angle to 0.5 for the whole folder" / "PD background 0.004" | `folder --clip-angle 0.5` / `folder --pd-background 0.004`. |
| "be stricter/looser about X" | `set <key> <value>` (keys in `rsoxr_reduce/settings.py`, e.g. `trim.snr_min`, `trim.min_overlap`, `qc.scale_outlier_min`, `qc.fringe_damping`); say which flags this changes, then `reduce`. |
| "reduce the energy scans" | `escan -p P` → `ReducedAuto/EnergyScans/*_Escan_th<θ>_<E0>-<E1>eV.dat` (÷ open beam point by point, PD background subtracted, gain-corrected). Plot them (dataviz conventions) and check the pre-edge is smooth. |
| "compare with the manual reduction" | `compare -p P "<raw>/Reduced"` — factor (absolute scale ratio) and shape differences per energy; `reduce --reference "<raw>/Reduced" --diagnostics all` overlays the manual curve on each diagnostic. On the reference calibration data (Nov 2025 carbon, see below) the factor was ≈ 0.08–0.12 because of the gain note there; other beamtimes may differ. |
| "now fit it" | hand off to the rsoxr-fit skill: `python -m rsoxr_harness init -p <fitproj> --data <raw>/ReducedAuto --file-type all ...`. |

## The rules (reason codes)

Pipeline order: **classify → select → trim → background → ÷ open beam → stitch → QC
→ write or hold**. Severity: *review* = energy held (no `.dat`) until accepted;
*info* = recorded only. Full thresholds and reasons: `REDUCTION_GUIDE.md`.

**1. Selection** (`selection.py`) — decided from the log + data, never the file name.
- `DUMMY/EMPTY/ZERO` (≤5 pts, max angle ≤1°, <200 B or no signal) — never used.
- `ENERGY_SCAN` (column 0 in eV) → `escan`. Also `OPEN_BEAM`, `ALIGNMENT`, `UNLOGGED`.
- Sections = clusters of detector + angular range (count time ignored), ordered by
  start angle (PD first).
- Per energy × section, **latest full-range scan wins** (`SUPERSEDED`; group convention: a re-measurement was re-measured for a reason).
- `PARTIAL_RANGE` — span < 80 % of nominal (review only if it ends early or leaves a gap).
- `OFF_SECTION` — range matches no section.
- `NO_PD` — energy rejected.
- `MISSING_SECTION` — middle section absent (review).
- `TRUNCATED_Q` — trailing section absent (info).
- `MIXED_SESSION` — chosen scans > 6 h apart (review).
- `POSITION_MISMATCH` — sections at different spots; info ≤ 3 mm (spots are stepped
  on purpose to limit beam damage).

**2. Trims** (`trim.py`)
- *PD start (beam clipping)* — one angle per folder: median over energies of the
  low-angle PD maximum.
  - `CLIP_SHIFT` — an energy's own peak is > 2 steps away.
  - `PD_SATURATION` — PD at the ADC limit (≥ 9.99 V) → start after the plateau
    (review if data above the clip angle are lost → lower the PD gain next time).
- *PD background* — one value per folder: intercept of a robust PD-vs-CEM
  regression in the overlap (PD dark offset), median over well-conditioned energies;
  fallback default per edge.
- *Junction ends* (PD→CT1, CTk→CTk+1) — keep the lower section into the overlap
  until the stitch scale is known to 2 % with ≥ 4 upper-section points, but not
  past its own noise floor (PD smoothed SNR < 5, CEM counts < 10).
- *CT start* — `CT_SATURATION`: CEM > 1e5 cps is non-linear (+3 % at 1–3e5,
  +22 % at 3e5–1e6 cps in the Nov 2025 reference data) → leading points trimmed.
- *Tail* — last section cut where counts < 10 (`LOW_COUNTS_TAIL`); final point always dropped.

**3. Stitch and open beam** (`stitch.py`, `pipeline.choose_open_beam`)
- Numerically identical to the widget/legacy path: trim → background → ÷ I0 →
  drop ≤0 → cumulative median-ratio stitch → Q with λ = 12399/E; flat 1 % errors.
- Open beam: same edge, then same filter, same exit-slit gap (±3 %), covering the
  energy, closest in time.
- **PD gain is in decades: I0 is multiplied by 10^(gain_sample − gain_open-beam)**
  (`GAIN_CORRECTED`); slit/filter mismatches are flagged, not corrected.

**4. QC** (`qc.py`)

Review holds:
- `NO_OVERLAP` — then `SCALE_FROM_GAP` (log-linear bridge of a ≤0.5° gap) or
  `SCALE_FROM_TREND` (nearest energies).
- `SCALE_OUTLIER` — gain-normalised scale > 30 % and > 3σ off a robust local line
  through energies within 6 eV (detector sensitivity has real structure at the C edge).
- `JUMP` — ratio drifts across the overlap.
- `FILTER_CHANGE` — filter/aperture differs between adjacent sections. Always review:
  beamline experience is that the higher-flux section can be over-exposed with
  damped fringes.
- `FRINGE_DAMPING` — upper/lower fringe amplitude < 0.75 with correlated fringes.
- `FRINGE_MISMATCH` — fringes don't correlate.

Info:
- `NOISY_OVERLAP`, `LOW_OVERLAP`.
- `SCALE_BLENDED` — a noisy overlap scale combined with the neighbour trend by inverse variance.
- `FRINGES_MATCH`.
- `FRINGE_CHECK_WEAK` — adjacent-window test when the overlap is too short (weak:
  same-filter junctions scatter 0.4–1.8).
- `NEG_AFTER_BG`, `TRUNCATED_Q`.

## Protocol

1. **Status first:** after every `reduce`, report ok / review / rejected counts and
   list held energies with codes. Nothing held is written to a `.dat`.
2. **Review = look:** before recommending `accept`, `Read` the diagnostic PNG and say
   what it shows (overlap agreement, fringe damping, kept ranges). For
   `FILTER_CHANGE`, compare fringe amplitude in the overlap panel; if damped, suggest
   dropping/trimming the damped section's overlap or accepting with a note — the
   user decides.
3. **Figures:** after reading a PNG, give the user a clickable link; they can't see
   what you Read.
4. **Overrides always carry the user's reason** (`--reason`); never accept on your own.
5. Respect shared-machine CPU limits, e.g. a 50 % CPU cap (everything here is
   single-process and light).
6. Report faithfully: when an automatic choice differs from an existing manual
   reduction (e.g. the Nov 2025 reference set), say so and why; don't silently match
   the old result.

## Reference calibration: Nov 2025 ALS beamtime (1530 manual energies recovered exactly)

The defaults in `settings.py` were tuned on this dataset. For other beamtimes, check
the dark offset and gains in the `reduce` report (and audit old reductions with
`calibrate`) rather than assuming these numbers.

- `recover.py` reconstructs any widget-made file exactly (trims, background, I0,
  scale) — use `calibrate --out truth.csv "<beamtime>"` to audit old reductions.
- Nov 2025 manual reductions used gain-8 open beams for gain-9 sample data → their R
  is ~10× (×1.2–1.3 more if the exit slit differed) too high: that is the "R ≈ 7 at
  low q" noted in rsoxr-fit for that data. ReducedAuto files are gain-corrected (R ≈ 0.7–1 at low q),
  so fitting scale starts near 1, not 9–10.
- Nov 2025 PD dark offsets: ≈0.0022–0.0028, but ≈0.006 in the Brewer9/10/13/15
  carbon session; manual backgrounds were often lower, which over-scales CT at
  absorbing energies where the PD/CT overlap sits on the PD floor.

## Tests

`$PY tests/test_reduce_harness.py` from `$REFLTOOLS` (plain asserts,
~1 min; synthetic beamtime + Nov 2025 legacy/recovery checks when the data exist —
the data path is set in `tests/test_reduce_harness.py`; those checks are skipped elsewhere).
