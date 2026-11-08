"""Subject-aware grading: region masks, region looks and person enhancement (Task G)."""

import colorsys

import json

import numpy as np

import pytest


from photo_grade import grade, measure, regions


from photo_grade.grade import apply

from photo_grade.looks import FINISH_DEFAULTS, LOOK_DEFAULTS, finish_errors

from photo_grade.regions import Subject, region_masks, upsample


from gradekit import NEUTRAL

FLAT_LOOK = {**LOOK_DEFAULTS, **NEUTRAL}

def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h / 360, s, v), np.float32)

def sky_over_sand(w=400, h=300, horizon=0.5):
    """Blue sky gradient (brighter towards the horizon) over orange sand."""
    img = np.empty((h, w, 3), np.float32)
    cut = int(h * horizon)
    for r in range(cut):
        img[r] = hsv(210, 0.55, 0.55 + 0.35 * r / cut)
    for r in range(cut, h):
        img[r] = hsv(35, 0.55, 0.75 - 0.2 * (r - cut) / (h - cut))
    return img

def full(m, img):
    return upsample(m, img.shape[:2])

def test_sky_and_sand_masks_on_synthetic_gradient():
    img = sky_over_sand()
    m = region_masks(img, Subject((400, 300)), ("sky", "sand"))
    sky, sand = full(m["sky"], img), full(m["sand"], img)
    assert sky.dtype == np.float32 and sky.shape == (300, 400)
    assert sky[:90].mean() > 0.95 and sky[210:].max() < 0.02
    assert sand[210:].mean() > 0.95 and sand[:90].max() < 0.02

def test_grey_sky_counts_as_sky():
    img = sky_over_sand()
    img[:150] = 0.85  # overcast: bright and colourless
    sky = full(region_masks(img, Subject((400, 300)), ("sky",))["sky"], img)
    assert sky[:60].mean() > 0.95

def test_masks_are_computed_on_a_1024_proxy_and_feathered():
    img = sky_over_sand(2048, 1536)
    m = region_masks(img, Subject((2048, 1536)), ("sky",))["sky"]
    assert max(m.shape) == regions.PROXY_SIDE
    col = m[:, m.shape[1] // 2]
    soft = ((col > 0.05) & (col < 0.95)).sum()
    sigma = regions.FEATHER * regions.PROXY_SIDE
    assert sigma * 1.5 < soft < sigma * 6, soft  # a soft edge about 2.5 sigma wide, not a hard step
    up = upsample(m, (1536, 2048))
    assert up.shape == (1536, 2048) and up.dtype == np.float32

def test_water_only_in_the_lower_70_percent_and_foliage_anywhere():
    img = np.empty((300, 400, 3), np.float32)
    img[:] = hsv(205, 0.5, 0.6)
    img[:, :100] = hsv(110, 0.5, 0.45)
    m = region_masks(img, Subject((400, 300)), ("water", "foliage"))
    water, foliage = full(m["water"], img), full(m["foliage"], img)
    assert water[220:, 200:].mean() > 0.95 and water[:60, 200:].max() < 0.02
    assert foliage[:, :70].mean() > 0.9 and foliage[:, 150:].max() < 0.02

def _food_scene():
    rng = np.random.default_rng(3)
    img = np.clip(0.45 + rng.normal(0, 0.02, (300, 400, 3)), 0, 1).astype(np.float32)
    yy, xx = np.mgrid[:300, :400]
    disc = (yy - 150) ** 2 + (xx - 200) ** 2 < 55 ** 2
    img[disc] = hsv(20, 0.75, 0.8)
    return img, disc

def test_food_is_the_centre_salient_region_only_when_categories_say_food():
    img, disc = _food_scene()
    food = full(region_masks(img, Subject((400, 300), label="food"), ("food",))["food"], img)
    assert food[disc].mean() > 0.6
    assert food[:40, :40].mean() < 0.1 and food[-40:, -40:].mean() < 0.1
    other = region_masks(img, Subject((400, 300), label="desert"), ("food",))["food"]
    assert other.max() == 0

def test_spectral_residual_saliency_peaks_on_the_odd_one_out():
    img, disc = _food_scene()
    s = regions.saliency(img @ grade.LUMA.astype(np.float32))
    s = upsample(s, img.shape[:2])
    assert s[disc].mean() > 3 * s[:60, :60].mean()

FACE = (160, 60, 240, 160)

def test_region_masks_never_touch_faces_or_the_person_below_them():
    img = np.empty((300, 400, 3), np.float32)
    img[:] = hsv(30, 0.55, 0.7)  # sand everywhere, the face is sand-coloured too
    subj = Subject((400, 300), faces=(FACE,), people=(FACE,))
    m = region_masks(img, subj, ("sand", "skin"))
    sand, skin = full(m["sand"], img), full(m["skin"], img)
    x0, y0, x1, y1 = FACE
    assert sand[y0:y1, x0:x1].max() == 0
    fh = y1 - y0
    assert sand[y1:min(300, int(y1 + 2.5 * fh)), x0:x1].max() == 0  # torso
    assert sand[250:, :40].mean() > 0.95 and sand[250:, 360:].mean() > 0.95
    assert skin[y0 + 20:y1 - 20, x0 + 20:x1 - 20].mean() > 0.9  # only the skin region may touch the face

def test_region_masks_scale_face_boxes_to_the_image():
    img = np.empty((150, 200, 3), np.float32)
    img[:] = hsv(35, 0.55, 0.7)
    sand = full(region_masks(img, Subject((400, 300), faces=(FACE,)), ("sand",))["sand"], img)
    assert sand[30:80, 80:120].max() == 0

def test_region_look_changes_only_its_region():
    img = sky_over_sand()
    out = apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None,
                regions={"sky": {"saturation": 1.3, "blue_lum": -0.06}}, subject=Subject((400, 300)))
    plain = apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None)
    assert np.abs(out[:100] - plain[:100]).mean() > 0.03
    assert np.abs(out[220:] - plain[220:]).max() < 1 / 255

def test_region_looks_start_from_neutral_not_the_defaults():
    img = sky_over_sand()
    out = apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None, regions={"sky": {}}, subject=Subject((400, 300)))
    assert np.abs(out - apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None)).max() < 1e-5

def test_regions_follow_the_global_look_order():
    img = sky_over_sand()
    look = {**FLAT_LOOK, "contrast": 0.3}
    out = apply(img, look, dict(FINISH_DEFAULTS), None, regions={"sand": {"orange_sat": 0.7}},
                subject=Subject((400, 300)))
    want = grade.look_transform(grade.look_transform(img[260:].astype(np.float64), **look),
                                **{**regions.REGION_NEUTRAL, "orange_sat": 0.7})
    assert np.abs(out[260:] - want).max() < 2e-3

def _portrait(h=400, w=600, seed=5):
    rng = np.random.default_rng(seed)
    img = np.clip(0.35 + rng.normal(0, 0.05, (h, w, 3)), 0, 1).astype(np.float32)
    img[...] *= np.array([1.0, 0.9, 0.8], np.float32)
    return img

PFACE = (260, 120, 340, 220)

PSIZE = (600, 400)

def test_person_enhancement_only_changes_pixels_near_the_face():
    img = _portrait()
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    person = {"exposure": 0.6, "contrast": 0.2, "clarity": 0.4, "skin_smooth": 0.4, "eyes": 1.0}
    out = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": person}, None, subject=subj)
    base = apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None)
    d = np.abs(out - base).max(-1)
    x0, y0, x1, y1 = PFACE
    assert d[y0:y1, x0:x1].mean() > 0.03
    far = np.ones_like(d, bool)
    fh, fw = y1 - y0, x1 - x0
    far[max(0, y0 - fh):, max(0, x0 - 2 * fw):x1 + 2 * fw] = False
    assert d[far].max() == 0
    assert out[y0:y1, x0:x1].mean() > base[y0:y1, x0:x1].mean() + 0.03  # the face is lifted

def test_person_defaults_and_no_people_are_no_ops():
    img = _portrait()
    base = apply(img, FLAT_LOOK, dict(FINISH_DEFAULTS), None)
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    assert np.array_equal(apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {}}, None, subject=subj), base)
    nobody = Subject(PSIZE, faces=(PFACE,), people=())
    assert np.array_equal(apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"exposure": 1.0}}, None,
                                subject=nobody), base)

def test_face_lift_brightens_shade_but_never_clips_a_sunlit_cheek():
    img = _portrait()
    x0, y0, x1, y1 = PFACE
    img[y0:y1, x0:(x0 + x1) // 2] = [0.93, 0.8, 0.7]  # sunlit half
    img[y0:y1, (x0 + x1) // 2:x1] = [0.4, 0.32, 0.28]  # shaded half
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    out = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"exposure": 1.0}}, None, subject=subj)
    face = out[y0:y1, x0:x1]
    assert face.max() < 0.99
    shade = (slice(y0 + 30, y1 - 30), slice((x0 + x1) // 2 + 10, x1 - 10))
    assert out[shade].mean() > img[shade].mean() * 1.25

def test_skin_smoothing_keeps_fine_texture_never_plastic():
    rng = np.random.default_rng(9)
    h = w = 400
    face = (100, 80, 300, 330)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    blotch = 0.04 * np.sin(xx / 9) * np.sin(yy / 11)  # uneven skin tone (mid frequencies)
    pores = rng.normal(0, 0.012, (h, w)).astype(np.float32)  # fine texture
    img = np.clip(hsv(22, 0.4, 0.7)[None, None] + (blotch + pores)[..., None], 0, 1).astype(np.float32)
    subj = Subject((w, h), faces=(face,), people=(face,))
    out = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"skin_smooth": 0.5}}, None, subject=subj)
    x0, y0, x1, y1 = face
    cheek = (slice(y0 + 140, y1 - 30), slice(x0 + 30, x1 - 30))
    lum = lambda a: a @ grade.LUMA.astype(np.float32)
    import cv2
    low = lambda a: cv2.GaussianBlur(lum(a), (0, 0), 3)
    high = lambda a: lum(a) - low(a)
    assert low(out)[cheek].std() < 0.85 * low(img)[cheek].std()  # blotches evened out
    assert high(out)[cheek].std() > 0.8 * high(img)[cheek].std()  # pores kept

def test_eye_sharpening_targets_the_eye_band():
    img = _portrait(seed=11)
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    out = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"eyes": 1.0}}, None, subject=subj)
    d = np.abs(out - img).max(-1)
    x0, y0, x1, y1 = PFACE
    fh = y1 - y0
    eyes = d[y0 + int(0.3 * fh):y0 + int(0.45 * fh), x0 + 15:x1 - 15].mean()
    chin = d[y1 - int(0.15 * fh):y1, x0 + 15:x1 - 15].mean()
    assert eyes > 0.01 and eyes > 4 * chin

def test_person_and_region_validation():
    assert finish_errors({"person": {"exposure": 0.5, "contrast": -0.3, "clarity": 0.5, "skin_smooth": 0.5,
                                     "eyes": 1}}) == []
    for bad in ({"person": {"exposure": 1.2}}, {"person": {"skin_smooth": 0.6}}, {"person": {"clarity": -0.1}},
                {"person": {"glow": 1}}, {"person": 0.5}, {"person": {"eyes": True}}):
        assert finish_errors(bad), bad
    assert regions.region_errors({"sky": {"saturation": 1.2}, "food": {}}) == []
    assert regions.region_errors({"moon": {}})
    assert regions.region_errors({"sky": {"saturation": 2}})
    assert regions.region_errors({"sky": 1})
    assert regions.region_errors([])

EMB = [1.0] + [0.0] * 511


def _eye_polys(cx, cy, w=14, h=6):
    def one(x):
        return tuple((x + dx, cy + dy) for dx in (-w, 0, w) for dy in (-h, h))
    return (one(cx - 20), one(cx + 20))

def test_measured_eye_outlines_replace_the_guessed_band():
    img = _portrait(seed=11)
    x0, y0, x1, y1 = PFACE
    fh = y1 - y0
    polys = _eye_polys(300, y0 + int(0.8 * fh))  # deliberately far from the default band
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,), people_eyes=(polys,))
    out = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"eyes": 1.0}}, None, subject=subj)
    d = np.abs(out - img).max(-1)
    exact = d[y0 + int(0.8 * fh) - 8:y0 + int(0.8 * fh) + 8, 260:340].mean()
    band = d[y0 + int(0.3 * fh):y0 + int(0.45 * fh), x0 + 15:x1 - 15].mean()
    assert exact > 0.01 and exact > 4 * band

def test_without_measured_eyes_the_band_is_still_used():
    img = _portrait(seed=11)
    plain = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    unmeasured = Subject(PSIZE, faces=(PFACE,), people=(PFACE,), people_eyes=(None,))
    a = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"eyes": 1.0}}, None, subject=plain)
    b = apply(img, FLAT_LOOK, {**FINISH_DEFAULTS, "person": {"eyes": 1.0}}, None, subject=unmeasured)
    assert np.array_equal(a, b)

def test_sky_and_water_only_overlap_between_35_and_40_percent():
    img = np.empty((1000, 300, 3), np.float32)  # tall, so a 5% band is ~3 feather sigmas
    img[:] = hsv(205, 0.5, 0.6)
    m = region_masks(img, Subject((300, 1000)), ("sky", "water"))
    sky, water = full(m["sky"], img)[:, 150], full(m["water"], img)[:, 150]
    assert sky[200] > 0.95 and sky[450] < 0.02
    assert water[300] < 0.02 and water[600] > 0.95

def _sunlit_portrait():
    img = _portrait()
    x0, y0, x1, y1 = PFACE
    img[y0:y1, x0:x1] = [0.93, 0.74, 0.6]  # sunlit skin: red already near the top
    return img

def _core(face=PFACE, k=1.0):
    x0, y0, x1, y1 = (v * k for v in face)
    fw, fh = x1 - x0, y1 - y0
    return (slice(int(y0 + 0.4 * fh), int(y0 + 0.7 * fh)), slice(int(x0 + 0.4 * fw), int(x0 + 0.6 * fw)))

WARM_LOOK = {**LOOK_DEFAULTS, **NEUTRAL, "warmth": 0.15, "skin_protect": 1.0}

def test_skin_protect_shields_the_face_core_from_the_warmth_term():
    img = _sunlit_portrait()
    subj = Subject(PSIZE, faces=(PFACE,), people=(PFACE,))
    plain = apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None)
    guarded = apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None, subject=subj)
    cold = apply(img, {**WARM_LOOK, "warmth": 0.0}, dict(FINISH_DEFAULTS), None)
    core = _core()
    assert plain[core][..., 0].max() > 0.99
    assert np.allclose(guarded[core][..., 0], cold[core][..., 0], atol=1e-4)
    assert guarded[core][..., 0].max() < 0.97
    far = (slice(-60, None), slice(0, 60))
    assert np.array_equal(guarded[far], plain[far])  # nothing outside the face changes

def test_warmth_guard_scales_with_skin_protect_and_needs_a_face():
    img = _sunlit_portrait()
    subj = Subject(PSIZE, faces=(PFACE,))
    core = _core()
    off = apply(img, {**WARM_LOOK, "skin_protect": 0.0}, dict(FINISH_DEFAULTS), None, subject=subj)
    plain = apply(img, {**WARM_LOOK, "skin_protect": 0.0}, dict(FINISH_DEFAULTS), None)
    assert np.array_equal(off, plain)
    half = apply(img, {**WARM_LOOK, "skin_protect": 0.5}, dict(FINISH_DEFAULTS), None, subject=subj)
    full_r = apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None, subject=subj)[core][..., 0].mean()
    assert full_r < half[core][..., 0].mean() <= apply(img, {**WARM_LOOK, "skin_protect": 0.5},
                                                       dict(FINISH_DEFAULTS), None)[core][..., 0].mean()
    nobody = apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None, subject=Subject(PSIZE))
    assert np.array_equal(nobody, apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None))

def test_warmth_guard_covers_only_the_chosen_people_when_people_are_chosen():
    img = _sunlit_portrait()
    other = (40, 120, 120, 220)
    img[120:220, 40:120] = [0.93, 0.74, 0.6]
    named = Subject(PSIZE, faces=(PFACE, other), people=(PFACE,), people_named=True)
    anyone = Subject(PSIZE, faces=(PFACE, other), people=(PFACE,))
    absent = Subject(PSIZE, faces=(PFACE, other), people=(), people_named=True)
    run = lambda s: apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None, subject=s)
    assert run(named)[_core()][..., 0].max() < 0.97 and run(named)[_core(other)][..., 0].max() > 0.99
    assert run(anyone)[_core(other)][..., 0].max() < 0.97
    assert np.array_equal(run(absent), apply(img, WARM_LOOK, dict(FINISH_DEFAULTS), None))
