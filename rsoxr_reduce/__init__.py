"""
rsoxr_reduce
------------
Plain-text driven reduction of soft X-ray reflectivity (RSoXR) raw data:
inventory -> failed-run selection -> rule-based trims -> stitch -> QC flags,
with a reason string for every decision.  Driven by the ``rsoxr-reduce`` skill
(.claude/skills/rsoxr-reduce/SKILL.md).  CLI: ``python -m rsoxr_reduce``.
"""
