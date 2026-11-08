import ast

import importlib.util

import json

import math

import os

import time

import tracemalloc

from pathlib import Path

import cv2

import numpy as np

import pytest

from photo_grade import grade, looks



from photo_grade.grade import GradeError, apply, match_params

from photo_grade.looks import FINISH_DEFAULTS, LOOK_DEFAULTS, finish_errors, look_errors

from gradekit import NEUTRAL, hsv_image





def test_finish_rejects_out_of_range():
    assert finish_errors(dict(FINISH_DEFAULTS)) == []
    assert finish_errors({"exposure": -1.5, "wb_r": 0.9, "wb_b": 1.1, "straighten": -10, "sharpen": 1.5,
                          "grain": 1}) == []
    for bad in ({"exposure": 1.6}, {"wb_r": 1.2}, {"wb_b": 0.85}, {"straighten": 10.5}, {"denoise": -0.1},
                {"clarity": 1.01}, {"sharpen": 1.6}, {"vignette": 2}, {"grain": 1.5}, {"grain": True},
                {"lens": 0.5}, {"sharpen": "0.3"}):
        assert finish_errors(bad), bad
    assert "unknown finish parameter" in finish_errors({"lens": 0.5})[0]

def _textured(h=90, w=120, seed=1):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    return np.clip(base * np.array([1.0, 0.8, 0.6], np.float32) + rng.normal(0, 0.03, (h, w, 3)), 0, 1).astype(
        np.float32)

def test_apply_identity_when_neutral():
    rgb = _textured()
    out = apply(rgb, {**LOOK_DEFAULTS, **NEUTRAL}, dict(FINISH_DEFAULTS), None)
    assert out.shape == rgb.shape and out.dtype == np.float32
    assert np.abs(out - rgb).max() < 1e-3

def test_apply_matches_look_transform_with_default_look():
    rgb = _textured()
    out = apply(rgb, dict(LOOK_DEFAULTS), dict(FINISH_DEFAULTS), None)
    ref = grade.look_transform(rgb.astype(np.float64), **LOOK_DEFAULTS)
    assert np.abs(out - ref).max() < 1e-5

def test_straighten_keeps_aspect_and_no_black_corners():
    rgb = np.full((300, 400, 3), 0.6, np.float32)
    for deg in (7.0, -10.0, 0.5):
        out = apply(rgb, {**LOOK_DEFAULTS, **NEUTRAL}, {**FINISH_DEFAULTS, "straighten": deg}, None)
        h, w = out.shape[:2]
        assert abs(w / h - 4 / 3) < 0.01
        t = math.radians(abs(deg))
        k = min(400 / (400 * math.cos(t) + 300 * math.sin(t)), 300 / (400 * math.sin(t) + 300 * math.cos(t)))
        assert 400 * k - 4 <= w <= 400 * k
        assert out.min() > 0.55

def test_exposure_and_wb_gains():
    rgb = np.full((20, 20, 3), 0.4, np.float32)
    neutral = {**LOOK_DEFAULTS, **NEUTRAL}
    out = apply(rgb, neutral, {**FINISH_DEFAULTS, "exposure": 1.0, "wb_r": 1.05, "wb_b": 0.95}, None)
    g = 2 ** (1 / 2.2)
    assert np.allclose(out[0, 0], [0.4 * g * 1.05, 0.4 * g, 0.4 * g * 0.95], atol=1e-4)

def test_finish_ops_change_the_image_sensibly():
    rgb = _textured(200, 300)
    neutral = {**LOOK_DEFAULTS, **NEUTRAL}
    sharp = apply(rgb, neutral, {**FINISH_DEFAULTS, "sharpen": 1.0}, None)
    assert np.abs(np.diff(sharp, axis=1)).mean() > np.abs(np.diff(rgb, axis=1)).mean()
    vig = apply(np.full((200, 300, 3), 0.5, np.float32), neutral, {**FINISH_DEFAULTS, "vignette": 1.0}, None)
    assert vig[0, 0].mean() < vig[100, 150].mean() - 0.1 and abs(vig[100, 150].mean() - 0.5) < 1e-3
    den = apply(rgb, neutral, {**FINISH_DEFAULTS, "denoise": 1.0}, None)
    assert np.abs(np.diff(den, axis=0)).mean() < np.abs(np.diff(rgb, axis=0)).mean()
    flat = np.full((200, 300, 3), 0.5, np.float32)
    grain = apply(flat, neutral, {**FINISH_DEFAULTS, "grain": 1.0}, None)
    assert 0.003 < grain.std() < 0.03
    assert np.array_equal(grain, apply(flat, neutral, {**FINISH_DEFAULTS, "grain": 1.0}, None))
    assert np.allclose(grain[..., 0], grain[..., 2])

def test_match_pulls_exposure_toward_ref():
    ref, shot = _textured(seed=2), _textured(seed=3) * 0.6
    m = match_params(ref, shot)
    assert m["exposure"] > 1
    out = apply(shot, {**LOOK_DEFAULTS, **NEUTRAL}, dict(FINISH_DEFAULTS), m)
    lum = lambda a: float((a @ grade.LUMA).mean())
    assert abs(lum(out) - lum(ref)) < abs(lum(shot) - lum(ref)) / 2


def test_straighten_geometry_identity_and_size():
    m, crop = grade.straighten_geometry(400, 300, 0.0)
    assert np.allclose(m, [[1, 0, 0], [0, 1, 0]]) and crop == (0, 0, 400, 300)
    assert grade.graded_size(400, 300, 0.0) == (400, 300)
    assert grade.map_box((10, 20, 110, 220), 400, 300, 0.0) == (10, 20, 110, 220)

def test_map_box_centre_stays_centred_and_corners_inside():
    w, h, deg = 400, 300, 5.0
    gw, gh = grade.graded_size(w, h, deg)
    x0, y0, x1, y1 = grade.map_box((180, 130, 220, 170), w, h, deg)
    assert abs((x0 + x1) / 2 - gw / 2) <= 1.5 and abs((y0 + y1) / 2 - gh / 2) <= 1.5
    assert 40 <= x1 - x0 <= 48  # rotated 40-px square's bounding box
    for box in ((0, 0, 80, 80), (320, 220, 400, 300), (0, 0, 400, 300)):
        got = grade.map_box(box, w, h, deg)
        assert got is not None and 0 <= got[0] < got[2] <= gw and 0 <= got[1] < got[3] <= gh
    assert grade.map_box((0, 0, 3, 3), w, h, 10.0) is None  # the corner is cropped away

def test_map_box_follows_the_pixels():
    w, h, deg = 400, 300, 6.0
    rgb = np.full((h, w, 3), 0.2, np.float32)
    rgb[100:140, 250:300] = 0.9
    out = apply(rgb, {**LOOK_DEFAULTS, **NEUTRAL}, {**FINISH_DEFAULTS, "straighten": deg}, None)
    ys, xs = np.nonzero(out[..., 0] > 0.55)
    x0, y0, x1, y1 = grade.map_box((250, 100, 300, 140), w, h, deg)
    assert abs(xs.min() - x0) <= 2 and abs(xs.max() + 1 - x1) <= 2
    assert abs(ys.min() - y0) <= 2 and abs(ys.max() + 1 - y1) <= 2
