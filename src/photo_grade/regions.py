"""Subject-aware grading: soft masks for a photo's main subject (sky, sand, water, foliage, food, skin), each
graded with its own look through look_transform, and local enhancement of the people in a photo.

Masks are worked out on a PROXY_SIDE copy and enlarged, so a 24 MP photo never holds more than one full-size
float32 mask at a time. grade.py imports this module lazily: it imports grade (look_transform) and measure
(the region HSV rules), and measure imports grade."""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .grade import _LUMA32, STRIP, _smoothstep, look_transform
from .looks import LOOK_DEFAULTS, PERSON_DEFAULTS, full_look_errors, split_look
from .measure import REGIONS, _mask, to_hsv

PROXY_SIDE = 1024
FEATHER = 0.015  # Gaussian sigma of a region mask's edge, as a fraction of the long side
REGION_NAMES = ("sky", "sand", "water", "foliage", "food", "skin")
# Sky and water are both blue; they share only rows 35-40% (the feathering blends the seam), so a sky look
# barely reaches the sea and the other way round. measure.REGIONS keeps its own 45% for its stats.
SKY_BOTTOM, WATER_TOP = 0.40, 0.35
# Same layout as measure.REGIONS: (hue deg, sat, value, rows, cols), ranges as fractions where relevant.
EXTRA_REGIONS = {
    "water": ((180, 230), (0.15, 1.0), (0.0, 1.0), (WATER_TOP, 1.0), (0.0, 1.0)),
    "foliage": ((70, 160), (0.15, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    # A sunset sky glows orange to magenta, which the blue sky rule misses; only used on sunset photos.
    "sky_warm": ((0, 60), (0.15, 1.0), (0.35, 1.0), (0.0, SKY_BOTTOM), (0.0, 1.0)),
    "sky_magenta": ((280, 360), (0.15, 1.0), (0.35, 1.0), (0.0, SKY_BOTTOM), (0.0, 1.0)),
}
SKIN_ANYWHERE = REGIONS["skin"][:3] + ((0.0, 1.0), (0.0, 1.0))
# The region a photo's subject label puts first.
SUBJECT_REGION = {"sunset": "sky", "desert": "sand", "beach": "water", "landscape": "foliage", "food": "food"}
# A region's look starts from no change at all; the global look has already been applied underneath it.
REGION_NEUTRAL = {**LOOK_DEFAULTS, "contrast": 0.0, "vibrance": 0.0, "warmth": 0.0}
# Spectral residual (Hou & Zhang 2007) on a thumbnail. 32 px rather than their 64: a dish fills much of a food
# photo, and the coarser map keeps the whole plate rather than just its rim.
SALIENCY_SIDE, SALIENCY_BLUR = 32, 1.5
CENTRE_SIGMA = 0.3  # centre weighting of the food mask, as a fraction of each side
FOOD_RAMP = (0.25, 0.5)  # normalised saliency that fades from "background" to "the dish"

# Person enhancement, all relative to the face height (fh) so it reads the same at any face size.
PERSON_FEATHER = 0.15  # sigma of the person mask's edge
TORSO_HEIGHT, TORSO_WIDTH = 2.5, 2.0  # x face height below the chin, x face width across
LIFT_KNEE = (0.55, 0.95)  # a face lift fades out between these brightest-channel values
TORSO_WEIGHT = 0.7  # the torso gets 70% of the face's lift at the shoulders, fading to 0 at its bottom
CLARITY_SIGMA = 0.08
SKIN_DETAIL = 1 / 120  # pores and fine texture finer than this (sigma, x fh) are never smoothed
SKIN_WORK = 200  # skin smoothing runs on the face at this height at most
SKIN_SPACE, SKIN_COLOUR = 0.06, 0.08  # bilateral sigmas: space x working face height, colour in luma
SKIN_MAX_SHIFT = 0.05  # no pixel's luma moves further than this: blemishes soften, skin never goes flat
SKIN_HSV = ((0, 50), (0.08, 0.75), (0.15, 1.0))
EYE_BAND = (0.2, 0.55, 0.08)  # rows (fractions of fh from the top of the box) and side inset (x fw)
EYE_SIGMA, EYE_AMOUNT = 0.008, 1.2
# The skin of the face for brightness checks: the box minus hair, ears and background.
FACE_CORE = (0.2, 0.2, 0.8, 0.9)
WARMTH_FEATHER = 0.05  # sigma of the warmth guard's edge, x face height
CLIP_LEVEL = 0.995

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Subject:
    """What a photo shows, in the graded file's px. faces: every face (kept out of every region but skin);
    people: the people to enhance and check; label: an optional subject label (e.g. "food", "sunset")."""
    size: tuple[int, int]
    faces: tuple[Box, ...] = ()
    people: tuple[Box, ...] = ()
    label: str | None = None
    # Parallel to `people`: that person's two eye outlines in the same px (None when not measured).
    people_eyes: tuple = ()
    # True when the people were chosen explicitly: `people` is then exactly them (possibly nobody).
    people_named: bool = False

    def guard_faces(self) -> tuple[Box, ...]:
        """The faces whose skin the warmth guard covers: the chosen people when there are any, every face otherwise."""
        return self.people if self.people_named else (self.faces or self.people)

    def scaled(self, shape, boxes) -> list[Box]:
        """boxes in the px of an image of this shape (H, W, ...)."""
        kx, ky = shape[1] / self.size[0], shape[0] / self.size[1]
        return [(b[0] * kx, b[1] * ky, b[2] * kx, b[3] * ky) for b in boxes]


def region_errors(regions) -> list[str]:
    if not isinstance(regions, dict):
        return [f"regions must be a dictionary of region name to look settings ({', '.join(REGION_NAMES)})"]
    out = []
    for name, look in regions.items():
        if name not in REGION_NAMES:
            out.append(f"unknown region {name!r} (allowed: {', '.join(REGION_NAMES)})")
        elif not isinstance(look, dict):
            out.append(f"region {name} must be a dictionary of look settings")
        else:
            out += [f"{name} {e}" for e in full_look_errors(look)]
    return out


def upsample(mask: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    if mask.shape[:2] == (h, w):
        return mask.astype(np.float32, copy=False)
    return cv2.resize(mask.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)


def _proxy(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = PROXY_SIDE / max(h, w)
    src = np.clip(rgb, 0, 1).astype(np.float32, copy=False) if k >= 1 else rgb
    if k >= 1:
        return src
    small = cv2.resize(np.ascontiguousarray(src, np.float32), (max(1, round(w * k)), max(1, round(h * k))),
                       interpolation=cv2.INTER_AREA)
    return np.clip(small, 0, 1)


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(m.astype(np.float32, copy=False), (0, 0), max(sigma, 0.5))


def _rule(hsv: np.ndarray, spec: tuple) -> np.ndarray:
    m, win = _mask(hsv, spec)
    out = np.zeros(hsv.shape[:2], np.float32)
    out[win] = m
    return out


def _rows(spec: tuple, bottom: float) -> tuple:
    return spec[:3] + ((spec[3][0], bottom), spec[4])


def person_boxes(face: Box, w: int, h: int) -> tuple[Box, Box]:
    """(head, torso) around a face box: the head takes in hair and chin; the torso hangs below the chin."""
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    head = (max(0.0, x0 - 0.2 * fw), max(0.0, y0 - 0.35 * fh), min(float(w), x1 + 0.2 * fw), min(float(h), y1 + 0.1 * fh))
    cx = (x0 + x1) / 2
    torso = (max(0.0, cx - TORSO_WIDTH / 2 * fw), min(float(h), y1),
             min(float(w), cx + TORSO_WIDTH / 2 * fw), min(float(h), y1 + TORSO_HEIGHT * fh))
    return head, torso


def _fill(m: np.ndarray, box: Box, value: float = 1.0) -> None:
    x0, y0, x1, y1 = (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))
    if x1 > x0 and y1 > y0:
        sub = m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        np.maximum(sub, value, out=sub)


def _people_hard(faces: list[Box], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    for f in faces:
        for box in person_boxes(f, w, h):
            _fill(m, box)
    return m


def _protect(faces: list[Box], h: int, w: int, sigma: float) -> np.ndarray:
    """1 on every face and the body below it (exactly), fading to 0 over a few sigma outside."""
    hard = _people_hard(faces, h, w)
    grow = max(1, round(2 * sigma))
    soft = _blur(cv2.dilate(hard, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)), sigma)
    return np.maximum(hard, soft)


def saliency(gray: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency (numpy FFT) of a luma image, 0..1, at SALIENCY_SIDE on the long side."""
    h, w = gray.shape[:2]
    k = SALIENCY_SIDE / max(h, w)
    small = cv2.resize(gray.astype(np.float32), (max(8, round(w * k)), max(8, round(h * k))),
                       interpolation=cv2.INTER_AREA).astype(np.float64)
    # Zero mean: otherwise the DC term comes back as a constant floor and everything looks salient.
    spec = np.fft.fft2(small - small.mean())
    log_amp = np.log(np.abs(spec) + 1e-9)
    log_amp[0, 0] = 0.0
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), SALIENCY_BLUR)
    return (sal / max(float(sal.max()), 1e-12)).astype(np.float32)


def _food(p: np.ndarray) -> np.ndarray:
    h, w = p.shape[:2]
    s = upsample(saliency(p @ _LUMA32), (h, w))
    yy = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) / CENTRE_SIGMA
    xx = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) / CENTRE_SIGMA
    s *= np.exp(-0.5 * (yy[:, None] ** 2 + xx[None, :] ** 2))
    s /= max(float(s.max()), 1e-12)
    lo, hi = FOOD_RAMP
    return np.clip((s - lo) / (hi - lo), 0, 1)


def _colour(hsv: np.ndarray, name: str, label: str | None) -> np.ndarray:
    if name == "sky":
        m = np.maximum(_rule(hsv, _rows(REGIONS["sky"], SKY_BOTTOM)), _rule(hsv, _rows(REGIONS["sky_grey"], SKY_BOTTOM)))
        if label == "sunset":
            m = np.maximum(m, np.maximum(_rule(hsv, EXTRA_REGIONS["sky_warm"]), _rule(hsv, EXTRA_REGIONS["sky_magenta"])))
        return m
    if name == "sand":
        return _rule(hsv, REGIONS["sand"])
    return _rule(hsv, EXTRA_REGIONS[name])


def region_masks(rgb: np.ndarray, subject: Subject, names) -> dict[str, np.ndarray]:
    """{region: soft 0..1 float32 mask} at proxy size (long side <= PROXY_SIDE); enlarge with upsample().
    Every region but skin is exactly 0 on each face and the body below it. "food" is all 0 unless the photo's
    subject category is food."""
    p = _proxy(rgb)
    ph, pw = p.shape[:2]
    hsv = to_hsv(p)
    sigma = FEATHER * max(ph, pw)
    faces = subject.scaled(p.shape, subject.faces)
    protect = _protect(faces, ph, pw, sigma) if faces else None
    out = {}
    for name in names:
        if name == "food":
            m = _blur(_food(p), sigma) if subject.label == "food" else np.zeros((ph, pw), np.float32)
        elif name == "skin":
            if faces:
                near = cv2.dilate(_people_hard(faces, ph, pw), np.ones((3, 3), np.uint8))
                m = _blur(_rule(hsv, SKIN_ANYWHERE) * near, sigma)
            else:
                m = _blur(_rule(hsv, REGIONS["skin"]), sigma)
        else:
            m = _blur(_colour(hsv, name, subject.label), sigma)
        if protect is not None and name != "skin":
            m *= 1 - protect
        out[name] = np.clip(m, 0, 1)
    return out


def active_regions(regions: dict | None) -> dict[str, dict]:
    """The regions whose look changes anything."""
    from .develop import ops_active
    return {k: v for k, v in (regions or {}).items()
            if {**REGION_NEUTRAL, **split_look(v)[0]} != REGION_NEUTRAL or ops_active(v)}


def apply_regions(x: np.ndarray, masks: dict[str, np.ndarray], looks: dict[str, dict],
                  people: np.ndarray | None = None) -> None:
    """Blend each region's look (look_transform, then its curve/hsl/grade) over x (float32, in place) through
    its mask, in row strips. people: the full-size people mask the look ops spare (develop.people_mask)."""
    from .develop import look_ops
    for name, look in looks.items():
        m = masks.get(name)
        if m is None or not m.any():
            continue
        params, ops = split_look({**REGION_NEUTRAL, **look})
        ops["skin_protect"] = params["skin_protect"]
        full = upsample(m, x.shape)
        for r in range(0, x.shape[0], STRIP):
            w = full[r:r + STRIP]
            if not w.any():
                continue
            xs = x[r:r + STRIP]
            ps = None if people is None else people[r:r + STRIP]
            xs += w[..., None] * (look_ops(look_transform(xs, **params).astype(np.float32), ops, ps) - xs)
        del full


# ---- person enhancement ----

def _soft_box(shape, box: Box, sigma: float) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    _fill(m, box)
    return _blur(m, sigma)


def _person_mask(shape, face: Box, sigma: float) -> np.ndarray:
    h, w = shape
    head, torso = person_boxes(face, w, h)
    m = np.zeros((h, w), np.float32)
    _fill(m, head)
    tx0, ty0, tx1, ty1 = (math.floor(v) for v in torso)
    if tx1 > tx0 and ty1 > ty0:
        fade = np.linspace(TORSO_WEIGHT, 0, ty1 - ty0, dtype=np.float32)[:, None]
        sub = m[ty0:ty1, tx0:tx1]
        np.maximum(sub, fade, out=sub)
    return _blur(m, sigma)


def _skin_smooth(sub: np.ndarray, face: Box, weight: np.ndarray, amount: float) -> None:
    """Frequency separation: detail finer than SKIN_DETAIL x fh is kept exactly; only the smooth base under it
    is evened out by an edge-preserving bilateral filter, and only on skin-coloured pixels."""
    x0, y0, x1, y1 = (max(0, math.floor(face[0])), max(0, math.floor(face[1])),
                      min(sub.shape[1], math.ceil(face[2])), min(sub.shape[0], math.ceil(face[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    fh = face[3] - face[1]
    region = sub[y0:y1, x0:x1]
    luma = region @ _LUMA32
    base = cv2.GaussianBlur(luma, (0, 0), max(0.5, SKIN_DETAIL * fh))
    k = min(1.0, SKIN_WORK / fh)
    small = cv2.resize(base, (max(4, round((x1 - x0) * k)), max(4, round((y1 - y0) * k))),
                       interpolation=cv2.INTER_AREA) if k < 1 else base
    even = cv2.bilateralFilter(small, 0, SKIN_COLOUR, max(1.0, SKIN_SPACE * fh * k))
    shift = upsample(even - small, base.shape)
    np.clip(shift, -SKIN_MAX_SHIFT, SKIN_MAX_SHIFT, out=shift)
    hsv = to_hsv(np.clip(region, 0, 1))
    (h0, h1), (s0, s1), (v0, v1) = SKIN_HSV
    skin = ((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= s0) & (hsv[..., 1] <= s1)
            & (hsv[..., 2] >= v0) & (hsv[..., 2] <= v1)).astype(np.float32)
    skin = _blur(skin, max(1.0, 0.01 * fh))
    region += (amount * skin * weight[y0:y1, x0:x1] * shift)[..., None]


def _eyes(sub: np.ndarray, face: Box, amount: float) -> None:
    """Unsharp mask on luma in the eye band. Face-detector boxes usually start at the forehead, so the eyes sit about
    20-55% down the box (the literal upper third would mostly sharpen forehead)."""
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    top, bottom, inset = EYE_BAND
    band = (x0 + inset * fw, y0 + top * fh, x1 - inset * fw, y0 + bottom * fh)
    pad = 0.1 * fh
    cx0, cy0 = max(0, math.floor(band[0] - pad)), max(0, math.floor(band[1] - pad))
    cx1, cy1 = min(sub.shape[1], math.ceil(band[2] + pad)), min(sub.shape[0], math.ceil(band[3] + pad))
    if cx1 - cx0 < 3 or cy1 - cy0 < 3:
        return
    region = sub[cy0:cy1, cx0:cx1]
    m = _soft_box(region.shape[:2], (band[0] - cx0, band[1] - cy0, band[2] - cx0, band[3] - cy0), 0.04 * fh)
    y = region @ _LUMA32
    y -= cv2.GaussianBlur(y, (0, 0), max(0.5, EYE_SIGMA * fh))
    region += (EYE_AMOUNT * amount * m * y)[..., None]


def _eyes_exact(sub: np.ndarray, polys, fh: float, amount: float) -> None:
    """Unsharp mask on luma inside the two measured eye outlines (grown a little, feathered)."""
    pts = np.concatenate([np.asarray(p, dtype=np.float32) for p in polys])
    pad = 0.15 * fh
    cx0, cy0 = max(0, math.floor(pts[:, 0].min() - pad)), max(0, math.floor(pts[:, 1].min() - pad))
    cx1, cy1 = min(sub.shape[1], math.ceil(pts[:, 0].max() + pad)), min(sub.shape[0], math.ceil(pts[:, 1].max() + pad))
    if cx1 - cx0 < 3 or cy1 - cy0 < 3:
        return
    region = sub[cy0:cy1, cx0:cx1]
    m = np.zeros(region.shape[:2], np.float32)
    for poly in polys:
        p = np.asarray(poly, dtype=np.float32) - (cx0, cy0)
        centre = p.mean(axis=0)
        hull = cv2.convexHull(((p - centre) * 1.3 + centre).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(m, hull, 1.0)
    m = _blur(m, max(0.5, 0.02 * fh))
    y = region @ _LUMA32
    y -= cv2.GaussianBlur(y, (0, 0), max(0.5, EYE_SIGMA * fh))
    region += (EYE_AMOUNT * amount * m * y)[..., None]


def enhance_person(x: np.ndarray, subject: Subject | None, person: dict | None) -> None:
    """Local face lift, gentle contrast, clarity, natural skin smoothing and sharper eyes for the subject's people,
    inside a feathered mask over each face and the torso below it. Pixels farther than 3 sigma from that mask
    are never touched. x: float32 HxWx3, in place."""
    p = {**PERSON_DEFAULTS, **(person or {})}
    if subject is None or not subject.people or not any(p.values()):
        return
    H, W = x.shape[:2]
    people = subject.scaled(x.shape, subject.people)
    kx, ky = x.shape[1] / subject.size[0], x.shape[0] / subject.size[1]
    eyes = [None if e is None else tuple(tuple((px * kx, py * ky) for px, py in poly) for poly in e)
            for e in (list(subject.people_eyes) + [None] * len(people))[:len(people)]]
    sigmas = [max(1.0, PERSON_FEATHER * (b[3] - b[1])) for b in people]
    pad = 3 * max(sigmas)
    outline = [box for f in people for box in person_boxes(f, W, H)]
    cx0, cy0 = max(0, math.floor(min(b[0] for b in outline) - pad)), max(0, math.floor(min(b[1] for b in outline) - pad))
    cx1, cy1 = min(W, math.ceil(max(b[2] for b in outline) + pad)), min(H, math.ceil(max(b[3] for b in outline) + pad))
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = x[cy0:cy1, cx0:cx1]
    local = [(b[0] - cx0, b[1] - cy0, b[2] - cx0, b[3] - cy0) for b in people]
    local_eyes = [None if e is None else tuple(tuple((px - cx0, py - cy0) for px, py in poly) for poly in e)
                  for e in eyes]
    m = np.zeros(sub.shape[:2], np.float32)
    for face, s in zip(local, sigmas):
        np.maximum(m, _person_mask(sub.shape[:2], face, s), out=m)
    if p["exposure"]:
        lift = m * (2 ** (p["exposure"] / 2.2) - 1)
        if p["exposure"] > 0:
            # A lift brightens the shaded skin and eases off towards white, so a sunlit cheek or forehead
            # doesn't clip (judged on the brightest channel, so red can't clip first). Monotonic up to +1.
            lift *= 1 - _smoothstep(LIFT_KNEE[0], LIFT_KNEE[1], sub.max(-1))
        sub *= (1 + lift)[..., None]
    if p["contrast"]:
        t = np.clip(sub, 0, 1)
        sub += p["contrast"] * m[..., None] * (t * t * (3 - 2 * t) - t)
    if p["clarity"]:
        fh = float(np.mean([f[3] - f[1] for f in local]))
        y = sub @ _LUMA32
        y -= cv2.GaussianBlur(y, (0, 0), max(1.0, CLARITY_SIGMA * fh))
        sub += (0.6 * p["clarity"] * m * y)[..., None]
    for face, polys in zip(local, local_eyes):
        if p["skin_smooth"]:
            _skin_smooth(sub, face, m, p["skin_smooth"])
        if p["eyes"]:
            if polys is None:
                _eyes(sub, face, p["eyes"])
            else:
                _eyes_exact(sub, polys, face[3] - face[1], p["eyes"])


def _core_box(box: Box, shape) -> tuple[int, int, int, int] | None:
    H, W = shape[:2]
    fw, fh = box[2] - box[0], box[3] - box[1]
    l, t, r, btm = FACE_CORE
    x0, y0 = max(0, math.floor(box[0] + l * fw)), max(0, math.floor(box[1] + t * fh))
    x1, y1 = min(W, math.ceil(box[0] + r * fw)), min(H, math.ceil(box[1] + btm * fh))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _guard_cores(subject: Subject, shape) -> list[tuple[Box, tuple[int, int, int, int]]]:
    return [(b, core) for b in subject.scaled(shape, subject.guard_faces())
            if (core := _core_box(b, shape)) is not None]


def warmth_guard_boxes(subject: Subject | None, shape, look: dict) -> list[tuple[int, int, int, int]]:
    """Padded boxes around each guarded face core (Subject.guard_faces) that the look's warmth term must not push
    towards clipping; empty when there is no warmth, no skin_protect or no such face. shape: the image (H, W, ...)."""
    if subject is None or not look.get("warmth") or not look.get("skin_protect"):
        return []
    H, W = shape[:2]
    out = []
    for b, core in _guard_cores(subject, shape):
        pad = math.ceil(3 * max(1.0, WARMTH_FEATHER * (b[3] - b[1])))
        out.append((max(0, core[0] - pad), max(0, core[1] - pad), min(W, core[2] + pad), min(H, core[3] + pad)))
    return out


def guard_skin_warmth(x: np.ndarray, subject: Subject, look: dict, before: list, boxes: list) -> None:
    """After the look: on each guarded face core hand back the red channel the look would have left without its
    warmth term, in proportion to skin_protect (feathered at the core's edge). `before`: copies of x's boxes
    taken before the look. Warmth adds red in highlights whatever the hue, so on sunlit skin it alone can clip red."""
    amount = float(look["skin_protect"])
    for (face, core), box, crop in zip(_guard_cores(subject, x.shape), boxes, before):
        sigma = max(1.0, WARMTH_FEATHER * (face[3] - face[1]))
        local = (core[0] - box[0], core[1] - box[1], core[2] - box[0], core[3] - box[1])
        w = _soft_box(crop.shape[:2], local, sigma) * amount
        cold = look_transform(crop, **{**look, "warmth": 0.0})
        sub = x[box[1]:box[3], box[0]:box[2]]
        sub[..., 0] += w * (cold[..., 0] - sub[..., 0])


# Final skin guard: red above the knee's start is eased towards its limit (never below the red the photo already
# had there), on the guarded face cores, after every grading op.
FINAL_SKIN_KNEE = (0.90, 0.96)
FINAL_SKIN_FULL = 0.5  # skin_protect at or above this guards fully; 0 turns the guard off


def final_skin_boxes(subject: Subject | None, shape) -> list[tuple[int, int, int, int]]:
    """Padded boxes around each guarded face core (Subject.guard_faces); empty without such a face."""
    if subject is None:
        return []
    H, W = shape[:2]
    out = []
    for b, core in _guard_cores(subject, shape):
        pad = math.ceil(4 * max(1.0, WARMTH_FEATHER * (b[3] - b[1])))
        out.append((max(0, core[0] - pad), max(0, core[1] - pad), min(W, core[2] + pad), min(H, core[3] + pad)))
    return out


def guard_final_skin(x: np.ndarray, subject: Subject, skin_protect: float, before: list, boxes: list) -> None:
    """After all grading: on each guarded face core, red the grade pushed past FINAL_SKIN_KNEE[0] is eased back
    along a smooth knee towards FINAL_SKIN_KNEE[1] (but never below the red that pixel had in `before`, the
    crops taken before grading), and green/blue are scaled by the same factor so the hue holds. Only warm
    pixels (red the top channel), feathered just outside the core."""
    amount = min(1.0, skin_protect / FINAL_SKIN_FULL)
    if amount <= 0:
        return
    k0, lim = FINAL_SKIN_KNEE
    for (face, core), box, crop in zip(_guard_cores(subject, x.shape), boxes, before):
        sub = x[box[1]:box[3], box[0]:box[2]]
        r = sub[..., 0]
        if not (r > k0).any():
            continue
        sigma = max(1.0, WARMTH_FEATHER * (face[3] - face[1]))
        g = 2 * sigma
        local = (core[0] - box[0] - g, core[1] - box[1] - g, core[2] - box[0] + g, core[3] - box[1] + g)
        w = _soft_box(sub.shape[:2], local, sigma) * amount
        w *= _smoothstep(-0.02, 0.02, r - np.maximum(sub[..., 1], sub[..., 2]))
        soft = np.where(r > k0, k0 + (lim - k0) * (1 - np.exp(-(r - k0) / (lim - k0))), r)
        target = np.maximum(soft, np.minimum(r, crop[..., 0]))
        new = r + w * (target - r)
        # Only pixels past the knee are touched; elsewhere (including dark or negative red) the factor is exactly 1.
        sub *= np.where(r > k0, new / np.maximum(r, k0), 1.0)[..., None]


def face_stats(rgb: np.ndarray, subject: Subject | None) -> list[dict]:
    """Per person: mean luma, the share of clipped pixels and the 99.5th percentile of the red channel
    on the skin of the face (FACE_CORE of the box), in rgb's px. "person" is the index into subject.people."""
    if subject is None:
        return []
    out = []
    for i, b in enumerate(subject.scaled(rgb.shape, subject.people)):
        box = _core_box(b, rgb.shape)
        if box is None:
            continue
        core = rgb[box[1]:box[3], box[0]:box[2]]
        out.append({"person": i, "luma": float((core @ _LUMA32).mean()), "clip": float((core.max(-1) >= CLIP_LEVEL).mean()),
                    "red_p995": float(np.percentile(core[..., 0], 99.5))})
    return out
