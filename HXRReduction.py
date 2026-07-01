import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg


class HXRProcessor:
    """
    Reduce Rigaku SmartLab hard x-ray reflectivity (.ras) data.

    Each .ras file may contain multiple scan segments stored as sequential
    *RAS_INT_START / *RAS_INT_END blocks with overlapping angle ranges.
    Segments are stitched by optimizing scale factors to minimise residuals
    in overlap regions (log-space least squares), replacing the need for
    hardcoded exposure-time scaling.

    Basic usage
    -----------
    proc = HXRProcessor()
    result = proc.reduce("sample.ras", footprint_angle=0.5)
    proc.plot(result)
    proc.save(result, "sample_reduced.dat")
    """

    def __init__(self):
        self.metadata = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def parse_ras_file(self, filepath):
        """
        Parse a Rigaku SmartLab .ras file.

        Parameters
        ----------
        filepath : str or Path

        Returns
        -------
        metadata : dict
            Header key/value pairs (global + per-block, later values win).
        segments : list of pd.DataFrame
            One DataFrame per *RAS_INT_START block, columns:
            ['angle', 'counts', 'attenuator'].
            DataFrame.attrs contains 'scan_speed', 'scan_start', 'scan_stop'.
        """
        filepath = Path(filepath)
        meta = {}
        all_blocks = []
        current_data = []
        block_meta_snapshot = {}
        in_data = False

        with open(filepath, "r", encoding="latin-1") as f:
            for line in f:
                line = line.rstrip("\n")

                if line.startswith("*RAS_INT_START"):
                    in_data = True
                    current_data = []
                    block_meta_snapshot = dict(meta)
                    continue

                if line.startswith("*RAS_INT_END"):
                    in_data = False
                    if current_data:
                        all_blocks.append((block_meta_snapshot, list(current_data)))
                    continue

                if in_data:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            angle = float(parts[0])
                            counts = float(parts[1])
                            attenuator = float(parts[2]) if len(parts) >= 3 else 1.0
                            current_data.append([angle, counts, attenuator])
                        except ValueError:
                            pass
                else:
                    if line.startswith("*") and " " in line:
                        key, _, value = line[1:].partition(" ")
                        meta[key] = value.strip().strip('"')

        segments = []
        for bm, data in all_blocks:
            arr = np.array(data)
            df = pd.DataFrame(
                {"angle": arr[:, 0], "counts": arr[:, 1], "attenuator": arr[:, 2]}
            )
            df.attrs["scan_speed"] = float(bm.get("MEAS_SCAN_SPEED", 1.0))
            df.attrs["scan_start"] = float(bm.get("MEAS_SCAN_START", 0.0))
            df.attrs["scan_stop"] = float(bm.get("MEAS_SCAN_STOP", 0.0))
            segments.append(df)

        if not segments:
            raise ValueError(f"No scan data found in {filepath}")

        return meta, segments

    def load_ras_files(self, filepaths):
        """
        Load one or more .ras files and return all scan segments.

        Parameters
        ----------
        filepaths : str, Path, or list of str/Path

        Returns
        -------
        metadata : dict
            Header metadata from the first file.
        segments : list of pd.DataFrame
            All scan segments across all files, in file order.
        """
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]

        all_segments = []
        metadata = None

        for fp in filepaths:
            meta, segs = self.parse_ras_file(fp)
            if metadata is None:
                metadata = meta
            all_segments.extend(segs)

        self.metadata = metadata
        return metadata, all_segments

    # ------------------------------------------------------------------
    # Q conversion and footprint
    # ------------------------------------------------------------------

    @staticmethod
    def angles_to_q(angles, wavelength=1.5406):
        """
        Convert TwoThetaTheta scan angles (2θ, degrees) to Q (Å⁻¹).

        In a coupled TwoThetaTheta scan the detector records 2θ; the
        sample angle is θ = 2θ/2.

        Q = 4π sin(θ) / λ  =  4π sin(angle/2) / λ
        """
        theta_rad = np.radians(np.asarray(angles, dtype=float) / 2.0)
        return 4.0 * np.pi * np.sin(theta_rad) / wavelength

    @staticmethod
    def footprint_correction(angles, footprint_angle, offset=0.0):
        """
        Geometric footprint correction factors.

        corr[i] = sin(angle[i] - offset) / sin(footprint_angle - offset)
                  for angle[i] < footprint_angle
        corr[i] = 1.0  for angle[i] >= footprint_angle

        Parameters
        ----------
        angles : array-like
            2θ scan angles (degrees) at which to evaluate correction.
        footprint_angle : float
            Reference 2θ angle (degrees) at which footprint = sample.
        offset : float
            Small geometric offset (degrees); was -0.01 in legacy notebook.

        Returns
        -------
        corr : np.ndarray, same shape as angles
        """
        angles = np.asarray(angles, dtype=float)
        corr = np.sin(np.radians(angles - offset))
        ref = np.sin(np.radians(footprint_angle - offset))
        corr /= ref
        corr[angles >= footprint_angle] = 1.0
        return corr

    # ------------------------------------------------------------------
    # Optimised stitching
    # ------------------------------------------------------------------

    def stitch_segments(self, seg_qs, seg_rs, min_overlap=3):
        """
        Find optimal scale factors by minimising log-space residuals in
        overlap regions (global least squares).

        S[0] = 1 (anchor).  For N-1 free parameters x = [log S[1], ...,
        log S[N-1]], each overlap pair (i, i+1) contributes equations:

            x[i] - x[i-1] = log( R_i(q_j) / R_{i+1}(q_j) )

        Solved with scipy.linalg.lstsq.

        Parameters
        ----------
        seg_qs : list of np.ndarray  — Q arrays per segment (Å⁻¹)
        seg_rs : list of np.ndarray  — R arrays per segment (counts)
        min_overlap : int            — minimum overlap points required

        Returns
        -------
        scales : np.ndarray, shape (N,)
        """
        N = len(seg_qs)
        if N == 1:
            return np.array([1.0])

        rows_A = []
        rows_b = []

        for i in range(N - 1):
            q_lo = max(seg_qs[i].min(), seg_qs[i + 1].min())
            q_hi = min(seg_qs[i].max(), seg_qs[i + 1].max())

            if q_lo >= q_hi:
                print(f"Warning: no Q overlap between segments {i} and {i+1}")
                continue

            mask = (seg_qs[i + 1] >= q_lo) & (seg_qs[i + 1] <= q_hi)
            q_ov = seg_qs[i + 1][mask]
            r_next = seg_rs[i + 1][mask]

            if mask.sum() < min_overlap:
                print(
                    f"Warning: only {mask.sum()} overlap points between "
                    f"segments {i} and {i+1} (need {min_overlap})"
                )
                continue

            r_prev_interp = np.interp(q_ov, seg_qs[i], seg_rs[i])

            valid = (r_prev_interp > 0) & (r_next > 0)
            if valid.sum() < min_overlap:
                continue

            log_ratios = np.log(r_prev_interp[valid] / r_next[valid])

            for lr in log_ratios:
                row = np.zeros(N - 1)
                if i > 0:
                    row[i - 1] = -1.0   # -log S[i]
                row[i] = 1.0            # +log S[i+1]
                rows_A.append(row)
                rows_b.append(lr)

        if not rows_A:
            print("Warning: no valid overlaps found; returning unit scale factors")
            return np.ones(N)

        A = np.array(rows_A)
        b = np.array(rows_b)
        x, _, _, _ = linalg.lstsq(A, b)

        scales = np.ones(N)
        scales[1:] = np.exp(x)
        return scales

    # ------------------------------------------------------------------
    # Main reduction pipeline
    # ------------------------------------------------------------------

    def reduce(
        self,
        filepaths,
        footprint_angle=None,
        footprint_offset=0.0,
        q_min=0.015,
        q_max=None,
        intensity_min=1e-4,
        wavelength=None,
        normalize=True,
        plot_diagnostics=False,
    ):
        """
        Full HXR reduction pipeline.

        Parameters
        ----------
        filepaths : str, Path, or list thereof
            One or more .ras files to load and stitch.
        footprint_angle : float, optional
            2θ reference angle (degrees) for footprint correction.
            No correction applied when None.
        footprint_offset : float
            Geometric offset for footprint formula (degrees).
        q_min : float
            Low-Q cutoff (Å⁻¹).
        q_max : float, optional
            High-Q cutoff (Å⁻¹).  No upper cut when None.
        intensity_min : float
            Minimum total_counts threshold (removes near-zero points).
        wavelength : float, optional
            Override wavelength (Å); uses header value if None.
        normalize : bool
            If True, divide R and dR by max(R) so R[0] ≤ 1.
        plot_diagnostics : bool
            If True, show a three-panel diagnostic plot: raw counts,
            footprint-corrected counts, and the correction function vs 2θ.

        Returns
        -------
        result : np.ndarray, shape (N, 3)
            Columns: [Q (Å⁻¹), R, dR]
        """
        metadata, segments = self.load_ras_files(filepaths)

        # Wavelength: header value (strip quotes) or override
        if wavelength is None:
            wl = float(metadata.get("HW_XG_WAVE_LENGTH_ALPHA1", 1.5406))
        else:
            wl = float(wavelength)

        print(f"Wavelength: {wl:.6f} Å")
        print(f"Loaded {len(segments)} scan segment(s)")

        seg_qs      = []
        seg_rs      = []   # footprint-corrected counts
        seg_rs_raw  = []   # raw counts (before footprint), for diagnostics
        seg_tc      = []   # total counts (before scaling), for uncertainty
        seg_angles  = []   # filtered 2θ angles, for diagnostics
        seg_corrs   = []   # per-point correction factors, for diagnostics

        for k, seg in enumerate(segments):
            angles = seg["angle"].values
            tc = seg["counts"].values * seg["attenuator"].values

            q = self.angles_to_q(angles, wl)

            # Intensity and Q filters
            mask = tc > intensity_min
            if q_min is not None:
                mask &= q > q_min
            if q_max is not None:
                mask &= q < q_max

            angles_f = angles[mask]
            tc_f     = tc[mask]
            q_f      = q[mask]

            if len(q_f) == 0:
                print(f"Segment {k}: no points after filtering — skipped")
                continue

            r_raw = tc_f.copy()
            r_f   = tc_f.copy()
            corr  = np.ones(len(tc_f))

            # Footprint correction
            if footprint_angle is not None:
                corr = self.footprint_correction(angles_f, footprint_angle, footprint_offset)
                r_f  = r_f / corr

            seg_qs.append(q_f)
            seg_rs.append(r_f)
            seg_rs_raw.append(r_raw)
            seg_tc.append(tc_f)
            seg_angles.append(angles_f)
            seg_corrs.append(corr)

        if not seg_qs:
            raise ValueError("No data remaining after filtering")

        # Sort segments by maximum Q so consecutive pairs always overlap
        # (needed when multiple files cover non-sequential angle ranges)
        order = np.argsort([q.max() for q in seg_qs])
        seg_qs     = [seg_qs[i]     for i in order]
        seg_rs     = [seg_rs[i]     for i in order]
        seg_rs_raw = [seg_rs_raw[i] for i in order]
        seg_tc     = [seg_tc[i]     for i in order]
        seg_angles = [seg_angles[i] for i in order]
        seg_corrs  = [seg_corrs[i]  for i in order]

        # Optimise scale factors
        scales = self.stitch_segments(seg_qs, seg_rs)
        for k, s in enumerate(scales):
            print(f"  Segment {k}: scale = {s:.4g}")

        # Assemble final arrays
        all_q, all_r, all_dr = [], [], []
        for k in range(len(seg_qs)):
            s = scales[k]
            all_q.append(seg_qs[k])
            all_r.append(seg_rs[k] * s)
            # Poisson uncertainty propagated through scale
            all_dr.append(np.sqrt(seg_tc[k]) * s)

        Q  = np.concatenate(all_q)
        R  = np.concatenate(all_r)
        dR = np.concatenate(all_dr)

        # Sort by Q
        idx = np.argsort(Q)
        Q, R, dR = Q[idx], R[idx], dR[idx]

        if normalize:
            norm = R.max()
            R  /= norm
            dR /= norm

        if plot_diagnostics:
            self._plot_footprint_diagnostics(
                seg_angles, seg_rs_raw, seg_rs, seg_corrs, scales,
                footprint_angle, footprint_offset,
            )

        return np.column_stack([Q, R, dR])

    def _plot_footprint_diagnostics(
        self, seg_angles, seg_rs_raw, seg_rs_corr, seg_corrs, scales,
        footprint_angle, footprint_offset,
    ):
        """
        Three-panel diagnostic plot showing, per scan segment:
          top    — raw stitched counts vs 2θ
          middle — footprint-corrected stitched counts vs 2θ
          bottom — footprint correction factor vs 2θ
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        ax_raw, ax_corr, ax_fp = axes

        all_angles = np.concatenate(seg_angles)

        for k, (ang, raw, corr_r, corr_f, s) in enumerate(
            zip(seg_angles, seg_rs_raw, seg_rs_corr, seg_corrs, scales)
        ):
            c = colors[k % len(colors)]
            lbl = f"seg {k}"
            ax_raw.semilogy(ang, raw * s,    ".", ms=2, color=c, label=lbl)
            ax_corr.semilogy(ang, corr_r * s, ".", ms=2, color=c, label=lbl)
            ax_fp.plot(ang, corr_f, "-", lw=1.2, color=c, label=lbl)

        for ax in (ax_raw, ax_corr):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, ncol=4, markerscale=3)

        ax_fp.grid(True, alpha=0.3)
        ax_fp.axhline(1.0, ls=":", color="k", lw=0.8)
        ax_fp.legend(fontsize=7, ncol=4)
        ax_fp.set_ylim(bottom=0)

        if footprint_angle is not None:
            for ax in axes:
                ax.axvline(footprint_angle, ls="--", color="k", lw=0.8,
                           label=f"footprint = {footprint_angle}°")
            # smooth correction curve over full angle range
            ang_dense = np.linspace(all_angles.min(), all_angles.max(), 800)
            fp_dense  = self.footprint_correction(ang_dense, footprint_angle, footprint_offset)
            ax_fp.plot(ang_dense, fp_dense, "k-", lw=1.5, label="correction curve", zorder=0)
            ax_fp.legend(fontsize=7, ncol=4)

        ax_raw.set_ylabel("Raw counts × scale")
        ax_corr.set_ylabel("Footprint-corrected × scale")
        ax_fp.set_ylabel("Correction factor")
        ax_fp.set_xlabel("2θ (degrees)")

        fig.suptitle("Footprint correction diagnostics", y=1.01)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def plot(self, result, title=None, ax=None, **kwargs):
        """
        Plot reduced reflectivity on a log-y scale with error bars.

        Parameters
        ----------
        result : np.ndarray, shape (N, 3)  — output of reduce()
        title  : str, optional
        ax     : matplotlib Axes, optional — created if not supplied

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        Q, R, dR = result[:, 0], result[:, 1], result[:, 2]
        ax.errorbar(
            Q, R, yerr=dR,
            fmt="o-", markersize=3, linewidth=0.8, capsize=2,
            **kwargs,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("Reflectivity")
        ax.grid(True, which="both", alpha=0.3)
        if title:
            ax.set_title(title)
        return ax

    def save(self, result, filepath, header="Q(Ang^-1)  R  dR"):
        """
        Save reduced data to a 3-column text file.

        Parameters
        ----------
        result   : np.ndarray, shape (N, 3)
        filepath : str or Path
        header   : str
        """
        np.savetxt(filepath, result, header=header, fmt="%.8e")
        print(f"Saved to {filepath}")
