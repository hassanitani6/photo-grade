import pytest

import numpy as np

from PIL import Image, ImageCms

import io

from pathlib import Path

from photo_grade.imageio import load_rgb, save_jpeg

from conftest import make_heic, make_jpeg  # noqa: E402

def test_load_rgb_applies_exif_orientation(tmp_path):
    p = make_jpeg(tmp_path / "a.jpg", (400, 200), orientation=6)
    assert load_rgb(p).shape == (400, 200, 3)

def test_load_rgb_reads_heic(tmp_path):
    arr = load_rgb(make_heic(tmp_path / "a.heic"))
    assert arr.shape == (200, 300, 3)
    assert arr.dtype == np.float32 and arr.max() <= 1.0

def test_load_rgb_max_side(tmp_path):
    arr = load_rgb(make_jpeg(tmp_path / "a.jpg", (400, 200)), max_side=100)
    assert arr.shape == (50, 100, 3)

P3 = "/System/Library/ColorSync/Profiles/Display P3.icc"

def test_load_rgb_converts_embedded_profile_to_srgb(tmp_path):
    import os
    import pytest
    if not os.path.exists(P3):
        pytest.skip("no Display P3 profile on this machine")
    p = tmp_path / "p3.png"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(p, icc_profile=Path(P3).read_bytes())
    got = load_rgb(p)[0, 0]
    # P3's pure green is outside sRGB: it clips to sRGB green, red goes to 0, not left as-is.
    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), (200, 100, 50)).save(plain)
    assert np.allclose(load_rgb(plain)[0, 0], np.array([200, 100, 50]) / 255)
    tagged = tmp_path / "mid.png"
    Image.new("RGB", (8, 8), (200, 100, 50)).save(tagged, icc_profile=Path(P3).read_bytes())
    r, g, b = load_rgb(tagged)[0, 0] * 255
    assert r > 205 and b < 48  # P3 orange is more saturated than the same numbers in sRGB
    assert got[1] > 0.95

def test_pixel_limit_allows_a_120mp_panorama_but_still_stops_bombs():
    import warnings

    from PIL import Image

    from photo_grade import imageio  # noqa: F401  (importing the reader sets the limit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Image._decompression_bomb_check((12000, 10000))
    with pytest.raises(Image.DecompressionBombError):
        Image._decompression_bomb_check((20000, 20000))
