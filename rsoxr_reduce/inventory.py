"""
inventory.py
------------
Read the beamline log and raw scan files and classify every file.

Filenames at the beamline are not reliable (``Brewer5_Ox_XT``, ``BrewerS9_f``,
``*_C_PD`` files that are really oxygen, one stem for PD and CT in 2026), so
everything here is decided from the log row (detector, energy, time, count
time) and the data themselves (column-0 range).  The filename is only used for
display and for its 6-digit counter.

Raw file format: one date line, then 4 tab-separated columns
(motor [deg or eV], signal, ~I0 channel, ring current).

Log format (ALS 11.0.1 / 6.3.2 style): tab separated, line 2 is the header,
data rows have one more field than the header, so we do not index by header
name.  Instead the energy is located as the value before the first 'eV'/'nm'
unit field, and the following (value, unit) pairs are sample θ, tilt,
azimuth, detector 2θ, then four translation motors (mm).
"""

import glob
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

RAW_RE = re.compile(r"^(?P<stem>.*?)(?P<counter>\d{6})\.(dat|txt)$")

EDGES = {  # name: (lo, hi) eV
    "C": (250.0, 360.0),
    "N": (380.0, 440.0),
    "O": (480.0, 620.0),
    "F": (660.0, 740.0),
}


def edge_of(energy):
    if energy is None:
        return None
    for k, (lo, hi) in EDGES.items():
        if lo <= energy <= hi:
            return k
    return None


# ----------------------------------------------------------------------
# log
# ----------------------------------------------------------------------
def _parse_dt(date, time):
    for fmt in ("%a, %b %d, %Y %I:%M:%S %p", "%a, %b %d, %Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date.strip()} {time.strip()}", fmt)
        except ValueError:
            continue
    return None


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_log_row(r):
    """One log row (list of fields) -> dict, or None if it is not a scan row."""
    if len(r) < 20 or not RAW_RE.match(r[2].strip()):
        return None
    out = {"filename": r[2].strip(), "comment": r[3].strip(),
           "datetime": _parse_dt(r[4], r[5]),
           "detector_name": r[6].strip().lower(),
           "gain_count": _num(r[8]), "scan_type": r[16].strip()}
    det = out["detector_name"]
    out["detector"] = "cem" if "cem" in det else ("pd" if "photo" in det else det or None)
    out["count_time"] = out["gain_count"] if out["detector"] == "cem" else None
    # PD amplifier gain is in decades (V/A = 10^gain); 'x10' adds one decade
    x10 = r[9].strip().upper() == "T"
    out["pd_gain"] = (out["gain_count"] + (1 if x10 else 0)) \
        if (out["detector"] == "pd" and out["gain_count"] is not None) else None
    # beamline optics (data fields sit one to the right of their header names)
    out["grating"] = _num(r[18])
    out["slit_gap"] = _num(r[19])
    out["filter"] = " ".join(x.strip() for x in r[21:23] if x.strip())
    # locate energy by its unit field
    iu = next((i for i in range(20, len(r)) if r[i].strip() in ("eV", "nm")), None)
    energy = None
    motors = []
    if iu is not None:
        v = _num(r[iu - 1])
        if v is not None:
            energy = v if r[iu].strip() == "eV" else (1239.84193 / v if v > 0 else None)
        j = iu + 1
        while j + 1 < len(r) + 1 and j < len(r):
            motors.append(_num(r[j]))
            j += 2
    out["energy"] = energy
    names = ["theta", "tilt", "azimuth", "two_theta", "m1", "x", "y", "z"]
    for n, v in zip(names, motors):
        out[n] = v
    for n in names[len(motors):]:
        out[n] = None
    return out


def parse_log(path):
    """filename -> log dict. Later duplicate rows (re-used names) win."""
    out = {}
    with open(path, encoding="latin1") as fh:
        for line in fh:
            d = parse_log_row(line.rstrip("\n").split("\t"))
            if d:
                out[d["filename"]] = d
    return out


def find_log(folder):
    """Walk up from ``folder`` to the first directory holding a *.log."""
    d = os.path.abspath(folder)
    for _ in range(6):
        logs = sorted(glob.glob(os.path.join(d, "*.log")))
        if logs:
            return logs[0]
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


# ----------------------------------------------------------------------
# raw files
# ----------------------------------------------------------------------
def read_raw(path):
    """-> (header_line, array Nx>=2). Empty/garbled files give a 0x4 array."""
    with open(path, encoding="latin1") as fh:
        header = fh.readline().strip()
    try:
        d = np.loadtxt(path, skiprows=1, ndmin=2)
    except Exception:
        d = np.zeros((0, 4))
    if d.size == 0:
        d = np.zeros((0, 4))
    return header, d


@dataclass
class FileInfo:
    name: str
    path: str
    stem: str
    counter: int
    size: int
    n: int = 0
    x0: float = None          # column-0 min
    x1: float = None          # column-0 max
    detector: str = None      # 'pd' | 'cem'
    count_time: float = None
    pd_gain: float = None
    energy: float = None
    edge: str = None
    datetime: str = None      # ISO
    x: float = None
    y: float = None
    slit_gap: float = None
    filter: str = None
    scan_type: str = None
    cls: str = None           # angle | energy_scan | open_beam | dummy | empty | alignment | unlogged
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @property
    def span(self):
        return (self.x1 - self.x0) if self.n else 0.0


def classify(fi, data, settings):
    """Set fi.cls / fi.reason from data + log fields."""
    s = settings["classify"]
    y = data[:, 1] if data.shape[0] else np.zeros(0)
    name = fi.name.lower()
    th, tth = fi.extra.get("theta"), fi.extra.get("two_theta")
    if fi.energy is None and fi.detector is None:
        fi.cls, fi.reason = "unlogged", "file not found in beamline log"
        return fi
    if fi.size < s["min_bytes"] or fi.n == 0:
        fi.cls, fi.reason = "empty", f"EMPTY: {fi.size} B, {fi.n} points"
        return fi
    if not np.any(y > 0):
        fi.cls, fi.reason = "empty", "ZERO: no positive intensity"
        return fi
    if fi.x1 > s["energy_min"] and fi.span > s["energy_scan_min_span"]:
        if "open" in name and "beam" in name:
            fi.cls, fi.reason = "open_beam", f"open-beam energy scan {fi.x0:.1f}-{fi.x1:.1f} eV"
        elif (th is not None and abs(th) < 0.05 and
              tth is not None and abs(tth) < 0.05):
            fi.cls, fi.reason = "open_beam", (f"energy scan at θ≈2θ≈0 "
                                              f"{fi.x0:.1f}-{fi.x1:.1f} eV")
        else:
            fi.cls, fi.reason = "energy_scan", (f"energy scan {fi.x0:.1f}-{fi.x1:.1f} eV "
                                                f"at θ={th}")
        return fi
    if fi.n <= s["dummy_max_points"] or fi.x1 <= s["dummy_max_angle"]:
        fi.cls, fi.reason = "dummy", (f"DUMMY: {fi.n} points, "
                                      f"{fi.x0:.2f}-{fi.x1:.2f}°")
        return fi
    if fi.x1 <= s["angle_max"] and fi.x0 >= -1.0 and fi.scan_type in ("script", None, ""):
        fi.cls, fi.reason = "angle", f"angle scan {fi.x0:.2f}-{fi.x1:.2f}° ({fi.n} pts)"
        return fi
    fi.cls, fi.reason = "alignment", (f"not a reflectivity scan: scan type "
                                      f"'{fi.scan_type}', {fi.x0:.2f}-{fi.x1:.2f}")
    return fi


def file_info(path, log, settings):
    """Build + classify the FileInfo for one raw file."""
    name = os.path.basename(path)
    m = RAW_RE.match(name)
    fi = FileInfo(name=name, path=os.path.abspath(path), stem=m["stem"] if m else name,
                  counter=int(m["counter"]) if m else -1, size=os.path.getsize(path))
    _, d = read_raw(path)
    fi.n = int(d.shape[0])
    if fi.n:
        fi.x0, fi.x1 = float(d[:, 0].min()), float(d[:, 0].max())
    L = log.get(name)
    if L:
        fi.detector, fi.count_time, fi.pd_gain = L["detector"], L["count_time"], L["pd_gain"]
        fi.energy = L["energy"]
        fi.datetime = L["datetime"].isoformat() if L["datetime"] else None
        fi.x, fi.y, fi.scan_type = L["x"], L["y"], L["scan_type"]
        fi.slit_gap, fi.filter = L["slit_gap"], L["filter"]
        fi.extra = {"theta": L["theta"], "two_theta": L["two_theta"]}
    e_for_edge = fi.energy
    if fi.n and fi.x1 > settings["classify"]["energy_min"]:
        e_for_edge = 0.5 * (fi.x0 + fi.x1)      # energy scan: use its own range
    fi.edge = edge_of(e_for_edge)
    classify(fi, d, settings)
    return fi


def inventory(folder, log, settings, recursive=False):
    """List + classify every raw file (6-digit counter) in ``folder``.

    ``log`` is a dict from :func:`parse_log`.  Returns list[FileInfo] sorted by
    acquisition time (log), then counter.
    """
    pattern = "**/*" if recursive else "*"
    out = []
    for p in sorted(glob.glob(os.path.join(folder, pattern), recursive=recursive)):
        if (os.path.isfile(p) and "Reduced" not in p
                and RAW_RE.match(os.path.basename(p))):
            out.append(file_info(p, log, settings))
    out.sort(key=lambda f: (f.datetime or "", f.counter, f.name))
    return out


def find_open_beams(paths, log, settings):
    """Open-beam FileInfos from files/dirs (non-recursive)."""
    out = []
    for p in paths:
        cands = [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "*")))
        for c in cands:
            if not os.path.isfile(c) or not RAW_RE.match(os.path.basename(c)):
                continue
            fi = file_info(c, log, settings)
            if fi.cls in ("open_beam", "energy_scan"):
                fi.cls = "open_beam"
                out.append(fi)
    return out
