"""Task W: dehaze, tone curve, HSL, colour-grading wheels, texture, local light, strength presets and the
strength sheet."""

import colorsys

import json

import time

import tracemalloc

import numpy as np

import pytest

from PIL import Image

from photo_grade import develop, grade, looks


from photo_grade.grade import _hue_deg, apply

from photo_grade.looks import FINISH_DEFAULTS, LOOK_DEFAULTS, PRESETS, finish_errors, full_look_errors

from photo_grade.regions import Subject

from gradekit import NEUTRAL, hsv_image

FLAT = {**LOOK_DEFAULTS, **NEUTRAL}

FACE = (260, 120, 340, 220)

SIZE = (600, 400)

def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h / 360, s, v), np.float32)

def ramp(h=60, w=400):
    g = np.linspace(0.02, 0.98, w, dtype=np.float32)
    return np.repeat(np.repeat(g[None, :, None], h, 0), 3, 2)

def textured(h=300, w=400, seed=3):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.15, 0.85, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    return np.clip(base * np.array([1.0, 0.85, 0.7], np.float32) + rng.normal(0, 0.04, (h, w, 3)), 0, 1).astype(
        np.float32)

def patches(hues, s=0.6, v=0.6, side=40):
    return np.concatenate([np.broadcast_to(hsv(h, s, v), (side, side, 3)) for h in hues], 1).copy()

def sat(x):
    return x.max(-1) - x.min(-1)

def run(img, look=None, finish=None, subject=None):
    return apply(img, {**FLAT, **(look or {})}, {**FINISH_DEFAULTS, **(finish or {})}, None, subject=subject)

def test_every_new_op_is_a_no_op_at_its_defaults():
    img = textured()
    base = run(img)
    look = {"curve": dict(looks.CURVE_DEFAULTS), "hsl": {b: {"h": 0, "s": 0, "l": 0} for b in looks.HSL_BANDS},
            "grade": {"shadows": {"hue": 200, "sat": 0}, "midtones": {"hue": 0, "sat": 0},
                      "highlights": {"hue": 40, "sat": 0}, "balance": 0.3, "blend": 0.7}}
    finish = {"dehaze": 0.0, "texture": 0.0, "light": {"subject": 0.0, "burn": 0.0}}
    subj = Subject((400, 300), faces=((150, 100, 220, 190),), people=((150, 100, 220, 190),))
    assert np.array_equal(run(img, look, finish, subj), base)
    assert np.array_equal(run(img, look, finish), base)

def test_look_defaults_and_ranges_list_the_same_keys():
    assert set(LOOK_DEFAULTS) == set(looks.LOOK_RANGES)
    assert not {"curve", "hsl", "grade", "preset"} & set(LOOK_DEFAULTS)

def test_curve_contrast_grows_with_strength_and_keeps_the_pivot():
    img = ramp()
    spreads = []
    for s in (0.0, 0.2, 0.4, 0.6):
        out = run(img, {"curve": {"strength": s, "pivot": 0.5}})
        y = out[0] @ grade._LUMA32
        assert np.all(np.diff(y) >= -1e-6)  # monotonic
        spreads.append(float(y[300] - y[100]))
        mid = int(np.argmin(np.abs(img[0, :, 0] - 0.5)))
        assert abs(y[mid] - 0.5) < 0.01
    assert spreads == sorted(spreads) and spreads[-1] > spreads[0] + 0.05

def test_curve_black_and_white_points():
    out = run(ramp(), {"curve": {"strength": 0.0, "black": 0.06, "white": 0.94}})
    x = ramp()[0, :, 0]
    y = out[0] @ grade._LUMA32
    assert np.all(y[x <= 0.06] < 1e-3)
    assert np.all(y[x >= 0.94] > 0.999)

def test_curve_spares_skin_by_skin_protect():
    skin, blue = hsv(22, 0.4, 0.35), hsv(215, 0.4, 0.35)
    img = np.concatenate([np.broadcast_to(skin, (40, 40, 3)), np.broadcast_to(blue, (40, 40, 3))], 1).copy()
    out = run(img, {"curve": {"strength": 0.6, "pivot": 0.55}, "skin_protect": 1.0})
    d_skin = np.abs(out[:, :40] - img[:, :40]).mean()
    d_blue = np.abs(out[:, 40:] - img[:, 40:]).mean()
    assert d_skin < 0.4 * d_blue

BANDS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 270,
         "magenta": 315}

def test_hsl_bands_are_isolated():
    img = patches(BANDS.values())
    for band, centre in BANDS.items():
        out = run(img, {"hsl": {band: {"s": 0.5, "l": -0.2}}, "skin_protect": 0.0})
        for k, (other, c) in enumerate(BANDS.items()):
            a, b = img[:, 40 * k:40 * k + 40], out[:, 40 * k:40 * k + 40]
            if other == band:
                assert sat(b).mean() > sat(a).mean() * 1.15, band
                assert (b @ grade._LUMA32).mean() < (a @ grade._LUMA32).mean() * 0.85, band
            elif min(abs(c - centre), 360 - abs(c - centre)) >= 60:
                assert np.abs(b - a).max() < 1e-4, (band, other)

def test_hsl_hue_shift_moves_hue_by_about_the_given_degrees():
    img = patches([200])
    for dh in (-12, 12):
        out = run(img, {"hsl": {"aqua": {"h": dh}, "blue": {"h": dh}}})
        moved = float(np.median(_hue_deg(out.astype(np.float64)) - _hue_deg(img.astype(np.float64))))
        assert abs(moved - dh) < 3

def test_hsl_leaves_grey_alone_and_spares_skin_but_not_sand():
    grey = np.full((20, 20, 3), 0.5, np.float32)
    assert np.allclose(run(grey, {"hsl": {"orange": {"s": 0.5, "l": 0.3}}}), grey, atol=1e-6)
    skin, sand = hsv(25, 0.35, 0.7), hsv(32, 0.75, 0.7)
    img = np.concatenate([np.broadcast_to(skin, (20, 20, 3)), np.broadcast_to(sand, (20, 20, 3))], 1).copy()
    out = run(img, {"hsl": {"orange": {"s": 0.3}}, "skin_protect": 1.0})
    assert sat(out[:, :20]).mean() < sat(img[:, :20]).mean() + 0.01
    assert sat(out[:, 20:]).mean() > sat(img[:, 20:]).mean() + 0.05

def test_split_tone_colours_shadows_and_highlights_separately():
    img = ramp()
    look = {"grade": {"shadows": {"hue": 200, "sat": 0.15}, "highlights": {"hue": 40, "sat": 0.15}},
            "skin_protect": 0.0}
    out = run(img, look)
    x = img[0, :, 0]
    dark, bright = out[0, (x > 0.15) & (x < 0.3)], out[0, (x > 0.7) & (x < 0.85)]
    assert np.all(dark[:, 2] > dark[:, 0] + 0.02)  # teal shadows
    assert np.all(bright[:, 0] > bright[:, 2] + 0.02)  # warm highlights
    assert np.allclose(out[0] @ grade._LUMA32, img[0] @ grade._LUMA32, atol=0.02)  # luma kept
    assert np.abs(out[0, x < 0.03] - img[0, x < 0.03]).max() < 0.01  # true black stays black
    shadows_only = run(img, {"grade": {"shadows": {"hue": 200, "sat": 0.15}}, "skin_protect": 0.0})
    assert np.abs(shadows_only[0, x > 0.85] - img[0, x > 0.85]).max() < 0.01

def test_balance_moves_the_split():
    img = ramp()
    x = img[0, :, 0]
    mid = (x > 0.45) & (x < 0.55)
    tone = {"shadows": {"hue": 200, "sat": 0.15}, "highlights": {"hue": 40, "sat": 0.15}}
    lo = run(img, {"grade": {**tone, "balance": -1.0}, "skin_protect": 0.0})[0, mid]
    hi = run(img, {"grade": {**tone, "balance": 1.0}, "skin_protect": 0.0})[0, mid]
    assert (hi[:, 0] - hi[:, 2]).mean() > (lo[:, 0] - lo[:, 2]).mean() + 0.02  # + balance favours highlights

def test_split_tone_spares_skin():
    skin = np.broadcast_to(hsv(22, 0.4, 0.3), (20, 20, 3)).copy()
    look = {"grade": {"shadows": {"hue": 190, "sat": 0.2}}}
    on = run(skin, {**look, "skin_protect": 0.0})
    off = run(skin, {**look, "skin_protect": 1.0})
    assert np.abs(off - skin).max() < 0.4 * np.abs(on - skin).max()

def hazy_scene(h=300, w=400, seed=4):
    """A detailed scene behind haze that thickens towards the top (the distance)."""
    rng = np.random.default_rng(seed)
    clear = np.clip(0.3 + 0.25 * rng.random((h // 10, w // 10, 3)), 0, 1).astype(np.float32)
    clear = np.kron(clear, np.ones((10, 10, 1), np.float32))
    clear *= np.array([0.9, 0.75, 0.55], np.float32)
    t = np.linspace(0.35, 0.9, h, dtype=np.float32)[:, None, None]
    air = np.array([0.88, 0.9, 0.92], np.float32)
    return clear * t + air * (1 - t)

def local_contrast(x):
    y = x @ grade._LUMA32
    return float(np.abs(np.diff(y, axis=1)).mean() + np.abs(np.diff(y, axis=0)).mean())

def test_dehaze_raises_local_contrast_on_a_hazy_scene_and_grows_with_strength():
    img = hazy_scene()
    c = [local_contrast(run(img, finish={"dehaze": d})) for d in (0.0, 0.3, 0.6, 1.0)]
    assert c == sorted(c) and c[2] > 1.3 * c[0]
    top = slice(0, 60)
    assert local_contrast(run(img, finish={"dehaze": 0.6})[top]) > 1.5 * local_contrast(img[top])

def test_dehaze_is_capped_at_30_percent_on_people():
    img = hazy_scene()
    subj = Subject((400, 300), faces=((170, 60, 230, 140),), people=((170, 60, 230, 140),))
    plain = run(img, finish={"dehaze": 1.0})
    guarded = run(img, finish={"dehaze": 1.0}, subject=subj)
    core = (slice(85, 125), slice(185, 215))
    full = np.abs(plain[core] - img[core]).mean()
    kept = np.abs(guarded[core] - img[core]).mean()
    assert kept <= 0.35 * full
    far = (slice(240, 300), slice(0, 60))
    assert np.allclose(guarded[far], plain[far], atol=1e-3)

def test_guided_filter_follows_edges():
    guide = np.zeros((64, 64), np.float32)
    guide[:, 32:] = 1.0
    noisy = guide + np.random.default_rng(0).normal(0, 0.2, guide.shape).astype(np.float32)
    out = develop.guided_filter(guide, noisy, 6, 1e-3)
    assert abs(out[:, :28].mean()) < 0.05 and abs(out[:, 36:].mean() - 1) < 0.05
    assert out[:, :28].std() < noisy[:, :28].std() / 3

def test_texture_adds_fine_detail_monotonically_and_less_on_people():
    img = textured()
    d = [local_contrast(run(img, finish={"texture": t})) for t in (0.0, 0.4, 0.8)]
    assert d == sorted(d) and d[2] > d[0] * 1.1
    face = (150, 100, 220, 190)
    subj = Subject((400, 300), faces=(face,), people=(face,))
    plain, guarded = run(img, finish={"texture": 0.8}), run(img, finish={"texture": 0.8}, subject=subj)
    core = (slice(120, 170), slice(165, 205))
    assert np.abs(guarded[core] - img[core]).mean() <= 0.35 * np.abs(plain[core] - img[core]).mean()

def scene_with_subject(h=300, w=400):
    img = np.full((h, w, 3), 0.4, np.float32) * np.array([1.0, 0.95, 0.9], np.float32)
    img[110:190, 150:250] = [0.5, 0.35, 0.25]  # the subject: a darker, warmer object in the middle
    return img

def test_light_brightens_the_subject_and_burns_the_edges_monotonically():
    img = scene_with_subject()
    centre, corner = (slice(130, 170), slice(180, 220)), (slice(0, 25), slice(0, 25))
    prev_c, prev_e = img[centre].mean(), img[corner].mean()
    for a in (0.15, 0.3, 0.5):
        out = run(img, finish={"light": {"subject": a, "burn": a}})
        assert out[centre].mean() > prev_c and out[corner].mean() < prev_e
        prev_c, prev_e = out[centre].mean(), out[corner].mean()
    assert prev_e < img[corner].mean() * 0.9

def test_burn_never_darkens_a_face_and_subject_light_never_clips():
    img = scene_with_subject()
    face = (10, 10, 70, 90)  # a face in the corner
    subj = Subject((400, 300), faces=(face,), people=(face,))
    out = run(img, finish={"light": {"burn": 0.5}}, subject=subj)
    core = (slice(35, 70), slice(25, 55))
    assert np.abs(out[core] - img[core]).max() < 0.01
    bright = np.full((300, 400, 3), 0.97, np.float32)
    assert run(bright, finish={"light": {"subject": 0.5}}).max() <= 0.99

def test_new_look_and_finish_keys_validate_with_ranges():
    good = {"curve": {"strength": 0.6, "black": 0.08, "white": 0.92, "pivot": 0.65},
            "hsl": {"blue": {"h": -15, "s": 0.5, "l": -0.3}, "orange": {"s": -0.5}},
            "grade": {"shadows": {"hue": 360, "sat": 0.25}, "midtones": {"hue": 0, "sat": 0},
                      "highlights": {"hue": 40, "sat": 0.1}, "balance": -1, "blend": 1}}
    assert full_look_errors({**good, "contrast": 0.2}) == []
    for bad in ({"curve": {"strength": 0.7}}, {"curve": {"white": 0.9}}, {"curve": {"pivot": 0.3}},
                {"curve": {"black": 0.1}}, {"curve": {"bend": 1}}, {"curve": 0.3},
                {"hsl": {"teal": {"s": 0.1}}}, {"hsl": {"blue": {"s": 0.6}}}, {"hsl": {"blue": {"h": 20}}},
                {"hsl": {"blue": {"l": 0.4}}}, {"hsl": {"blue": {"x": 0}}}, {"hsl": {"blue": 0.1}},
                {"grade": {"shadows": {"sat": 0.3}}}, {"grade": {"shadows": {"hue": 400}}},
                {"grade": {"balance": 2}}, {"grade": {"blend": -0.1}}, {"grade": {"lows": {}}},
                {"contrast": 0.9}, {"preset": "punchy"}):
        assert full_look_errors(bad), bad
    assert finish_errors({"dehaze": 1.0, "texture": 1.0, "light": {"subject": 0.5, "burn": 0.5}}) == []
    for bad in ({"dehaze": 1.1}, {"texture": -0.1}, {"light": {"subject": 0.6}}, {"light": {"burn": -1}},
                {"light": {"dodge": 0.1}}, {"light": 0.2}):
        assert finish_errors(bad), bad

def test_presets_validate_and_are_ordered_by_strength():
    assert list(PRESETS) == ["natural", "punchy", "bold"]
    for name, p in PRESETS.items():
        assert full_look_errors(p["look"]) == [], name
        assert finish_errors(p["finish"]) == [], name
    n, p, b = (PRESETS[k] for k in ("natural", "punchy", "bold"))
    for get in (lambda x: x["look"]["curve"]["strength"], lambda x: x["finish"]["dehaze"],
                lambda x: x["finish"]["texture"], lambda x: x["finish"]["light"]["subject"],
                lambda x: x["finish"]["light"]["burn"], lambda x: x["look"]["grade"]["highlights"]["sat"],
                lambda x: x["look"]["grade"]["shadows"]["sat"], lambda x: x["look"]["hsl"]["orange"]["s"],
                lambda x: -x["look"]["hsl"]["blue"]["l"]):
        assert get(n) < get(p) < get(b)


def sunlit_face(h=400, w=600):
    rng = np.random.default_rng(5)
    img = np.clip(0.35 + rng.normal(0, 0.05, (h, w, 3)), 0, 1).astype(np.float32) * np.array([1.0, 0.9, 0.8],
                                                                                              np.float32)
    x0, y0, x1, y1 = FACE
    img[y0:y1, x0:x1] = np.clip(np.array([0.93, 0.74, 0.6], np.float32)
                                + rng.normal(0, 0.015, (y1 - y0, x1 - x0, 3)).astype(np.float32), 0, 1)
    return img

def face_core(face=FACE):
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    return (slice(int(y0 + 0.2 * fh), int(y0 + 0.9 * fh)), slice(int(x0 + 0.2 * fw), int(x0 + 0.8 * fw)))

PUNCHY_SUNLIT = {**LOOK_DEFAULTS, "contrast": 0.14, "vibrance": 0.3, "warmth": 0.03, "skin_protect": 0.9,
                 **PRESETS["punchy"]["look"]}

def test_punchy_never_pushes_sunlit_skin_red_past_0_97():
    img = sunlit_face()
    subj = Subject(SIZE, faces=(FACE,), people=(FACE,), people_named=True)
    finish = {**FINISH_DEFAULTS, **PRESETS["punchy"]["finish"], "person": {"exposure": 0.3, "eyes": 0.4}}
    out = apply(img, PUNCHY_SUNLIT, finish, None, subject=subj)
    assert np.percentile(out[face_core()][..., 0], 99.5) <= 0.97
    unguarded = apply(img, PUNCHY_SUNLIT, finish, None, subject=subj, final_skin_guard=False)
    assert np.percentile(unguarded[face_core()][..., 0], 99.5) > 0.97  # the guard is what holds it
    # hue stays natural: red/green ratio barely moves where the guard acts
    core = face_core()
    ratio = lambda a: np.median(a[core][..., 1] / np.maximum(a[core][..., 0], 1e-4))
    assert abs(ratio(out) - ratio(unguarded)) < 0.02

def test_final_skin_guard_leaves_non_face_pixels_alone():
    img = sunlit_face()
    subj = Subject(SIZE, faces=(FACE,), people=(FACE,), people_named=True)
    finish = {**FINISH_DEFAULTS, **PRESETS["punchy"]["finish"]}
    on = apply(img, PUNCHY_SUNLIT, finish, None, subject=subj)
    off = apply(img, PUNCHY_SUNLIT, finish, None, subject=subj, final_skin_guard=False)
    x0, y0, x1, y1 = FACE
    outside = np.ones(on.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(on[outside], off[outside])

def test_light_subject_stays_off_faces():
    img = scene_with_subject()
    face = (170, 110, 230, 190)  # a face in the middle of the subject
    subj = Subject((400, 300), faces=(face,), people=(face,))
    out = run(img, finish={"light": {"subject": 0.5}}, subject=subj)
    core = (slice(130, 170), slice(185, 215))
    assert np.abs(out[core] - img[core]).max() < 0.01

def clear_scene(h=300, w=400):
    img = np.empty((h, w, 3), np.float32)
    for r in range(h // 2):
        img[r] = hsv(212, 0.6 - 0.25 * r / (h // 2), 0.85)
    rng = np.random.default_rng(2)
    ground = np.kron(rng.random((h // 20, w // 10, 1)).astype(np.float32), np.ones((10, 10, 1), np.float32))
    img[h // 2:] = (0.1 + 0.4 * ground[: h - h // 2]) * np.array([0.55, 0.4, 0.15], np.float32)
    return img

def test_haze_factor_is_near_zero_on_clear_air_and_high_in_haze():
    assert develop.haze_factor(clear_scene()) < 0.1
    assert develop.haze_factor(hazy_scene()) > 0.7
    walls = np.full((300, 400, 3), 0.92, np.float32)  # a white wall indoors
    walls[200:] = [0.35, 0.2, 0.1]
    assert develop.haze_factor(walls) < 0.1

def test_region_looks_spare_people():
    img = np.full((300, 400, 3), 0.0, np.float32)
    img[:] = hsv(22, 0.4, 0.5)
    face = (170, 60, 230, 140)
    subj = Subject((400, 300), faces=(face,), people=(face,))
    look = {"curve": {"strength": 0.6, "pivot": 0.65}, "skin_protect": 1.0}
    out = apply(img, FLAT, dict(FINISH_DEFAULTS), None, regions={"skin": look}, subject=subj)
    core = (slice(85, 125), slice(185, 215))
    assert np.abs(out[core] - img[core]).max() < 0.01

def test_dehaze_keeps_headroom_for_the_look():
    img = hazy_scene()
    img[:20, :20] = [1.0, 0.95, 0.5]  # a bright warm patch, brighter in red than the haze's airlight
    out = img.copy()
    develop.dehaze(out, 1.0, None)
    assert out.max() > 1.0  # values above white survive until the final clip

def test_preset_errors_are_plain_for_any_value():
    for bad in (["punchy"], {"a": 1}, 3, "loud"):
        errs = looks.preset_errors(bad)
        assert errs and "natural, punchy, bold" in errs[0]

def test_final_skin_guard_never_touches_dark_or_negative_pixels():
    img = sunlit_face()
    x0, y0, x1, y1 = FACE
    img[y0 + 60:y0 + 80, x0 + 30:x0 + 50] = [0.01, 0.01, 0.03]  # a dark beard patch inside the core
    subj = Subject(SIZE, faces=(FACE,), people=(FACE,), people_named=True)
    look = {**PUNCHY_SUNLIT, "curve": {"strength": 0.6, "black": 0.08, "pivot": 0.5}}
    finish = {**FINISH_DEFAULTS, "person": {"exposure": 0.45, "clarity": 0.3}}
    on = apply(img, look, finish, None, subject=subj)
    off = apply(img, look, finish, None, subject=subj, final_skin_guard=False)
    patch = (slice(y0 + 60, y0 + 80), slice(x0 + 30, x0 + 50))
    assert np.array_equal(on[patch], off[patch])


def test_presets_carry_a_full_base_look_and_finish():
    for name, p in PRESETS.items():
        assert {"contrast", "vibrance", "highlights"} <= set(p["look"]), name
        assert {"sharpen", "clarity"} <= set(p["finish"]), name
    n, pu, b = (PRESETS[k] for k in ("natural", "punchy", "bold"))
    for key in ("contrast", "vibrance"):
        assert n["look"][key] < pu["look"][key] < b["look"][key]
    assert n["finish"]["clarity"] < pu["finish"]["clarity"] < b["finish"]["clarity"]

def test_clarity_is_halved_on_people():
    img = textured()
    face = (150, 100, 220, 190)
    subj = Subject((400, 300), faces=(face,), people=(face,))
    plain, guarded = run(img, finish={"clarity": 0.8}), run(img, finish={"clarity": 0.8}, subject=subj)
    core = (slice(120, 170), slice(165, 205))
    ratio = np.abs(guarded[core] - img[core]).mean() / np.abs(plain[core] - img[core]).mean()
    assert 0.4 < ratio < 0.6
    far = (slice(0, 40), slice(0, 40))
    assert np.allclose(guarded[far], plain[far], atol=1e-6)

def test_presets_keep_detail_gentle_so_night_and_action_camera_shots_dont_turn_grainy():
    caps = {"natural": {"sharpen": 0.2, "texture": 0.1, "clarity": 0.15},
            "punchy": {"sharpen": 0.2, "texture": 0.1, "clarity": 0.15},
            "bold": {"sharpen": 0.25, "texture": 0.15, "clarity": 0.2}}
    for name, p in PRESETS.items():
        for key, cap in caps[name].items():
            assert p["finish"].get(key, 0.0) <= cap, (name, key)
        assert p["finish"].get("denoise", 0.0) == 0.0, name  # denoise paints watercolour skies
    n, pu, b = (PRESETS[k]["finish"] for k in ("natural", "punchy", "bold"))
    for key in ("sharpen", "texture", "clarity"):
        assert n[key] <= pu[key] <= b[key], key
