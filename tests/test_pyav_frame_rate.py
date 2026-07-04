"""PyAV frame rate coercion for video encode."""

from __future__ import annotations

from fractions import Fraction

from ltx_media import _pyav_frame_rate


def test_pyav_frame_rate_from_float():
    assert _pyav_frame_rate(24.0) == Fraction(24, 1)


def test_pyav_frame_rate_from_int():
    assert _pyav_frame_rate(24) == Fraction(24, 1)


def test_pyav_frame_rate_fraction_passthrough():
    frac = Fraction(24000, 1001)
    assert _pyav_frame_rate(frac) is frac
