"""REPORT.md, the committed output of the full run, against the paper's findings."""

from __future__ import annotations

import re

import pytest


def value(report: str, name: str) -> float:
    match = re.search(rf"- {re.escape(name)}: \*\*(-?[\d.]+)\*\*", report)
    assert match, f"{name} missing from REPORT.md"
    return float(match.group(1))


def test_recogniser_works(report, targets):
    match = re.search(r"\| lfw \| ([\d.]+)% \|", report)
    assert match
    assert float(match.group(1)) >= targets["clean_lfw_accuracy_floor"]


@pytest.mark.parametrize("disturbance", ["blur", "center occlusion"])
def test_quality_falls_when_the_face_is_degraded(report, disturbance):
    """Sec. 5: occlusions and other ICAO incompliances lower the estimated quality."""
    assert value(report, f"Drop under {disturbance}") > 0


def test_quality_is_in_the_unit_range(report):
    """Eqs. (1)-(2): the scalar quality is bounded by [0, 1]."""
    assert 0.0 <= value(report, "QI clean") <= 1.0
