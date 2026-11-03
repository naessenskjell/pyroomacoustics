"""
Tests for the ASF broadband air absorption (constants asf_fast_air_abs and
asf_air_abs_coeff), used for ultrasonic simulations.
"""
import numpy as np
import pytest

import pyroomacoustics as pra

room_dim = [8.0, 4.0, 2.4]
src = [1.0, 1.0, 1.0]
mic = [6.0, 3.0, 1.2]
fs = 1000000


def make_rir(materials, air_absorption, fast):
    pra.constants.set("asf_fast_air_abs", fast)
    try:
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=materials,
            max_order=0,
            air_absorption=air_absorption,
        )
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        return room.rir[0][0], room.c
    finally:
        pra.constants.set("asf_fast_air_abs", False)


def test_asf_air_abs_single_band_direct_path():
    # direct path only: the peak is scaled by exp(-0.5 * a * d)
    h_air, _ = make_rir(pra.Material(0.45), True, True)
    h_ref, _ = make_rir(pra.Material(0.45), False, True)
    d = np.linalg.norm(np.array(src) - np.array(mic))
    expected = np.exp(-0.5 * pra.constants.get("asf_air_abs_coeff") * d)
    ratio = np.max(np.abs(h_air)) / np.max(np.abs(h_ref))
    assert ratio == pytest.approx(expected, rel=1e-5)


def test_asf_air_abs_stays_single_band():
    h_air, _ = make_rir(pra.Material(0.45), True, True)
    h_ref, _ = make_rir(pra.Material(0.45), False, True)
    # no octave-band filtering: same shape, only scaled
    n = min(len(h_air), len(h_ref))
    corr = np.dot(h_air[:n], h_ref[:n]) / np.linalg.norm(h_air[:n]) / np.linalg.norm(h_ref[:n])
    assert corr == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("materials", ["wood_techtile", "panel_fabric_covered_6pcf"])
def test_asf_fast_octave_bands_equivalent(materials):
    # the fast multi-band RIR construction equals the upstream band loop
    rirs = []
    for fast in (False, True):
        pra.constants.set("asf_fast_octave_bands", fast)
        try:
            room = pra.ShoeBox(
                room_dim,
                fs=fs,
                materials=pra.Material(materials),
                max_order=3,
            )
            room.add_source(src)
            room.add_microphone(mic)
            room.compute_rir()
            rirs.append(room.rir[0][0])
        finally:
            pra.constants.set("asf_fast_octave_bands", False)
    assert len(rirs[0]) == len(rirs[1])
    err = np.linalg.norm(rirs[1] - rirs[0]) / np.linalg.norm(rirs[0])
    assert err < 1e-6


def test_asf_air_abs_multi_band_unchanged():
    # as in the original ASF patch, multi-band rooms get no air absorption
    materials = pra.Material("wood_techtile")
    h_air, _ = make_rir(materials, True, True)
    h_ref, _ = make_rir(materials, False, True)
    n = min(len(h_air), len(h_ref))
    # atol: float noise from the multi-threaded RIR builder near zero crossings
    np.testing.assert_allclose(h_air[:n], h_ref[:n], atol=1e-12 * np.abs(h_ref).max())
