"""Develop ops, for punch that stays realistic: dehaze, texture and local light (finish) and a filmic tone
curve, per-hue HSL and colour-grading wheels (look). Order inside grade.apply follows the usual photo-editing
order: white balance -> exposure -> dehaze / clarity / texture -> look -> curve -> HSL -> colour grading ->
region looks -> person -> local light.

Every op is a no-op at its defaults. Masks (dehaze transmission, people, light) are worked out on a
regions.PROXY_SIDE copy and enlarged, so a 24 MP photo holds at most one or two full-size float32 maps."""
import math

import cv2
import numpy as np

from .grade import _LUMA32, SKIN_HUE, SKIN_WIDTH, STRIP, _hue_deg, _smoothstep
from .looks import CURVE_DEFAULTS, GRADE_DEFAULTS, HSL_BAND_DEFAULTS, HSL_BANDS, LIGHT_DEFAULTS, look_with_ops
from .regions import FEATHER, LIFT_KNEE, PROXY_SIDE, Subject, _blur, _food, _protect, _proxy, upsample

# Skin is spared from the look ops by hue (SKIN_HUE +- SKIN_WIDTH, as look_transform does) and HSV saturation:
# full from SKIN_SAT_IN up, fading out over SKIN_SAT_OUT, where saturated desert sand (same hue) begins.
SKIN_SAT_IN = (0.08, 0.2)
SKIN_SAT_OUT = (0.55, 0.75)
HSL_SAT_KNEE = (0.6, 0.95)
HSL_CENTRES = (0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 270.0, 315.0)  # in HSL_BANDS order
SHADOW_FLOOR = (0.02, 0.15)
WHEEL_SAT_KNEE = (0.25, 0.6)  # the shadow wheel fades in over these lumas, so true black never takes a colour cast

DEHAZE_OMEGA = 0.95  # He et al.: keep a trace of haze so distance still reads
DEHAZE_PATCH = 0.015  # dark-channel patch, x the proxy's long side
DEHAZE_RADIUS, DEHAZE_EPS = 0.03, 1e-3  # guided filter radius (x long side) and regulariser
DEHAZE_FLOOR = 0.25  # transmission floor: never amplify contrast more than 4x
DEHAZE_ON_PEOPLE = 0.3  # at most this share of the dehaze strength on faces and the body below them
DEHAZE_CONTRAST = 0.05  # S-curve added back per unit of dehaze
HAZE_BRIGHT = 0.8  # haze_factor ignores lumas above this
HAZE_RAMP = (0.12, 0.35)  # grey-weighted dark-channel median that reads as clear .. thick haze
TEXTURE_SIGMA, TEXTURE_GAIN = 0.004, 0.6  # x long side; amount per unit
CLARITY_ON_PEOPLE = 0.5  # clarity ages skin: faces get half
TEXTURE_ON_PEOPLE = 0.15  # texture ages skin and lifts red speculars on sunlit faces
SUBJECT_STOPS, BURN_STOPS = 1.0, 1.2  # at light 1.0 (the range stops at 0.5)
LIGHT_BLUR = 0.04  # sigma of the subject mask, x long side
BURN_START = 0.3  # burn starts this far out (0 centre, 1 corner)


# ---- look ops ----

def _skin_weight(x: np.ndarray, skin_protect: float) -> np.ndarray:
    mx = x.max(-1)
    s = (mx - x.min(-1)) / np.maximum(mx, 1e-4)
    d = np.abs(((_hue_deg(x) - SKIN_HUE + 180) % 360) - 180)
    band = np.clip(1 - d / SKIN_WIDTH, 0, 1)
    return (skin_protect * band * _smoothstep(*SKIN_SAT_IN, s) * (1 - _smoothstep(*SKIN_SAT_OUT, s))).astype(
        np.float32)


def _curve_on(c: dict) -> bool:
    return c["strength"] > 0 or c["black"] > 0 or c["white"] < 1


def _hsl_on(h: dict) -> bool:
    return any(any(v[k] for k in HSL_BAND_DEFAULTS) for v in h.values())


def _grade_on(g: dict) -> bool:
    return any(g[w]["sat"] > 0 for w in ("shadows", "midtones", "highlights"))


def ops_active(look: dict) -> bool:
    """Whether the look's curve, hsl or grade changes anything."""
    full = look_with_ops(look)
    return _curve_on(full["curve"]) or _hsl_on(full["hsl"]) or _grade_on(full["grade"])


def curve_luma(y: np.ndarray, c: dict) -> np.ndarray:
    """Black/white points, then a filmic S about the pivot (power 2 toe and shoulder) blended in by strength."""
    t = np.clip((y - c["black"]) / (c["white"] - c["black"]), 0, 1)
    s, p = c["strength"], c["pivot"]
    if s:
        sc = np.where(t < p, p * (t / p) ** 2, 1 - (1 - p) * ((1 - t) / (1 - p)) ** 2)
        t = t + s * (sc - t)
    return t.astype(np.float32)


def _curve(x: np.ndarray, c: dict) -> np.ndarray:
    # The curve moves luma; colours scale with it so hue and saturation hold. Where scaling would clip a channel,
    # the rest comes as grey, so bright colours roll off towards white instead of changing hue.
    y = x @ _LUMA32
    t = curve_luma(y, c)
    ratio = t / np.maximum(y, 1e-4)
    r = np.minimum(ratio, 1 / np.maximum(x.max(-1), 1e-4))
    return x * r[..., None] + (t - y * r)[..., None]


def band_weights(hue: np.ndarray) -> list[np.ndarray]:
    """One weight per HSL band (HSL_BANDS order), smoothstep cross-fades between neighbouring centres; they sum
    to 1 at every hue."""
    out = []
    n = len(HSL_CENTRES)
    for i, c in enumerate(HSL_CENTRES):
        nxt = (HSL_CENTRES[(i + 1) % n] - c) % 360
        prv = (c - HSL_CENTRES[i - 1]) % 360
        d = ((hue - c + 180) % 360) - 180
        t = np.where(d >= 0, d / nxt, -d / prv)
        out.append((1 - _smoothstep(0, 1, t)).astype(np.float32))
    return out


def _hsl(x: np.ndarray, bands: dict) -> np.ndarray:
    hue = _hue_deg(x)
    chroma = np.clip((x.max(-1) - x.min(-1)) * 4, 0, 1)
    dh = np.zeros(x.shape[:2], np.float32)
    ds, dl = np.zeros_like(dh), np.zeros_like(dh)
    for name, w in zip(HSL_BANDS, band_weights(hue)):
        b = bands[name]
        if not any(b[k] for k in HSL_BAND_DEFAULTS):
            continue
        w = w * chroma
        dh += w * b["h"]
        ds += w * b["s"]
        dl += w * b["l"]
    y0 = x @ _LUMA32
    if dh.any():
        # Rotate the colour about the grey axis: _hue_deg moves by exactly dh. Luma is put back afterwards.
        th = np.radians(dh)[..., None]
        m = x.mean(-1, keepdims=True)
        v = x - m
        cross = np.stack([v[..., 2] - v[..., 1], v[..., 0] - v[..., 2], v[..., 1] - v[..., 0]], -1) / math.sqrt(3)
        x = m + v * np.cos(th) + cross * np.sin(th)
        x = x + (y0 - x @ _LUMA32)[..., None]
    if ds.any():
        # A saturation boost eases off on colour that's already near full saturation (HSV S from HSL_SAT_KNEE),
        # so a pushed band can't turn spices or tiles into radioactive orange or neon cyan; cuts apply fully.
        mx = x.max(-1)
        hs = (mx - x.min(-1)) / np.maximum(mx, 1e-4)
        ds = np.where(ds > 0, ds * (1 - _smoothstep(*HSL_SAT_KNEE, hs)), ds)
        x = y0[..., None] + (x - y0[..., None]) * (1 + ds)[..., None]
    if dl.any():
        x = x * (1 + dl)[..., None]
    return x


def wheel_vector(hue: float) -> np.ndarray:
    """A luma-neutral RGB offset of that hue (in _hue_deg's angles) whose max-minus-min is about 1."""
    c, s = math.cos(math.radians(hue)), math.sin(math.radians(hue))
    v = np.array([2 * c / 3, -c / 3 + s / math.sqrt(3), -c / 3 - s / math.sqrt(3)], np.float32)
    return v - float(v @ _LUMA32)


def _wheels(x: np.ndarray, g: dict) -> np.ndarray:
    y = x @ _LUMA32
    split = 0.5 - 0.25 * g["balance"]  # + balance: the highlight colour reaches further down
    width = 0.1 + 0.3 * g["blend"]
    t = _smoothstep(split - width, split + width, y)
    weights = {"shadows": (1 - t) * _smoothstep(*SHADOW_FLOOR, y), "highlights": t,
               "midtones": np.clip(1 - np.abs(y - split) / (2 * width), 0, 1)}
    # Split toning colours the neutral and muted tones; already colourful pixels take less of it, so a teal shadow
    # can't push a dark blue-green into neon cyan (HSV saturation soars when colour is added to dark pixels).
    mx = x.max(-1)
    muted = 1 - _smoothstep(*WHEEL_SAT_KNEE, (mx - x.min(-1)) / np.maximum(mx, 1e-4))
    out = x.copy()
    for name, w in weights.items():
        w = w * muted
        wh = g[name]
        if wh["sat"] > 0:
            out += (wh["sat"] * w)[..., None] * wheel_vector(wh["hue"])
    return out


def look_ops(x: np.ndarray, look: dict, people: np.ndarray | None = None) -> np.ndarray:
    """Curve -> HSL -> colour wheels on a float32 strip, spared on skin and on the people mask in proportion to
    skin_protect. Returns x itself when nothing is set."""
    full = look_with_ops(look)
    on = (_curve_on(full["curve"]), _hsl_on(full["hsl"]), _grade_on(full["grade"]))
    if not any(on):
        return x
    sp = float(full.get("skin_protect", 0.0))
    spare = _skin_weight(x, sp) if sp else None
    if spare is not None and people is not None:
        np.maximum(spare, sp * people, out=spare)
    y = x
    if on[0]:
        y = _curve(y, full["curve"])
    if on[1]:
        y = _hsl(y, full["hsl"])
    if on[2]:
        y = _wheels(y, full["grade"])
    y = y.astype(np.float32, copy=False)
    if spare is None:
        return y
    return x + (1 - spare)[..., None] * (y - x)


def people_mask(subject: Subject | None, shape) -> np.ndarray | None:
    """Faces and the body below them, 1 inside and feathered out, at proxy size (None without faces)."""
    if subject is None or not subject.faces:
        return None
    h, w = shape[:2]
    k = min(1.0, PROXY_SIDE / max(h, w))
    ph, pw = max(1, round(h * k)), max(1, round(w * k))
    faces = subject.scaled((ph, pw), subject.faces)
    return _protect(faces, ph, pw, FEATHER * max(ph, pw))


# ---- finish ops ----

def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter with box means (edge-preserving smoothing of src along guide's edges)."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda a: cv2.boxFilter(a, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    g, s = guide.astype(np.float32), src.astype(np.float32)
    mg, ms = mean(g), mean(s)
    a = (mean(g * s) - mg * ms) / (mean(g * g) - mg * mg + eps)
    b = ms - a * mg
    return mean(a) * g + mean(b)


def transmission(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(atmospheric light A (3,), refined transmission at p's size) by the dark-channel prior."""
    long = max(p.shape[:2])
    k = max(3, round(DEHAZE_PATCH * long)) | 1
    kernel = np.ones((k, k), np.uint8)
    dark = cv2.erode(p.min(-1), kernel)
    n = max(1, dark.size // 1000)  # the brightest 0.1% of the dark channel
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    a = np.clip(p.reshape(-1, 3)[idx].mean(0), 0.05, 1.0).astype(np.float32)
    t = 1 - DEHAZE_OMEGA * cv2.erode((p / a).min(-1).astype(np.float32), kernel)
    t = guided_filter(p @ _LUMA32, t, max(2, round(DEHAZE_RADIUS * long)), DEHAZE_EPS)
    return a, np.clip(t, 0, 1)


def haze_factor(rgb: np.ndarray) -> float:
    """0 (clear air) .. 1 (thick haze): the median dark channel of the photo's midtones, weighted by how grey
    they are. Sky (blue), white walls and highlights (luma > HAZE_BRIGHT) are left out: the dark-channel prior
    reads them as haze, which is why an unscaled dehaze darkens clear skies and white walls."""
    p = _proxy(rgb)
    long = max(p.shape[:2])
    k = max(3, round(DEHAZE_PATCH * long)) | 1
    dark = cv2.erode(p.min(-1), np.ones((k, k), np.uint8))
    mx = p.max(-1)
    s = (mx - p.min(-1)) / np.maximum(mx, 1e-4)
    hue = _hue_deg(p)
    sky = (hue >= 180) & (hue <= 250) & (s > 0.15)
    use = ((p @ _LUMA32) <= HAZE_BRIGHT) & ~sky
    if use.mean() < 0.05:
        return 0.0
    veil = float(np.median((dark * (1 - s))[use]))
    return float(_smoothstep(*HAZE_RAMP, np.array(veil)))


def dehaze(x: np.ndarray, amount: float, subject: Subject | None) -> None:
    """In place on float32 x: J = (I - A) / t + A with t from the proxy, strength `amount`, then a little contrast
    back. On people (faces and the body below them) at most DEHAZE_ON_PEOPLE of the strength."""
    p = _proxy(x)
    a, t = transmission(p)
    s = np.full(t.shape, amount, np.float32)
    guard = people_mask(subject, x.shape)
    if guard is not None:
        s *= 1 - (1 - DEHAZE_ON_PEOPLE) * guard
    teff = np.maximum(1 - s * (1 - t), DEHAZE_FLOOR)
    tf = upsample(teff, x.shape)
    sf = upsample(s, x.shape) if guard is not None else None
    for r in range(0, x.shape[0], STRIP):
        xs = x[r:r + STRIP]
        xs -= a
        xs /= tf[r:r + STRIP][..., None]
        xs += a
        # No clip here: values past white keep their headroom for the look's highlight roll-off.
        t = np.clip(xs, 0, 1)
        k = DEHAZE_CONTRAST * (sf[r:r + STRIP][..., None] if sf is not None else amount)
        xs += k * (t * t * (3 - 2 * t) - t)


def clarity(x: np.ndarray, amount: float, subject: Subject | None) -> None:
    """Local contrast: luma unsharp, radius 1% of the long side, amount 0.6 x clarity; half strength on people."""
    y = x @ _LUMA32
    y -= cv2.GaussianBlur(y, (0, 0), 0.01 * max(x.shape[:2]))
    y *= 0.6 * amount
    guard = people_mask(subject, x.shape)
    if guard is not None:
        y *= upsample(1 - (1 - CLARITY_ON_PEOPLE) * guard, x.shape)
    x += y[..., None]


def texture(x: np.ndarray, amount: float, subject: Subject | None) -> None:
    """Fine-detail luma unsharp (sigma TEXTURE_SIGMA x long side), separate from clarity; reduced on people."""
    y = x @ _LUMA32
    y -= cv2.GaussianBlur(y, (0, 0), max(0.5, TEXTURE_SIGMA * max(x.shape[:2])))
    y *= TEXTURE_GAIN * amount
    guard = people_mask(subject, x.shape)
    if guard is not None:
        y *= upsample(1 - (1 - TEXTURE_ON_PEOPLE) * guard, x.shape)
    x += y[..., None]


def light_stops(shape, light: dict, subject: Subject | None, p: np.ndarray) -> np.ndarray:
    """Stops of local light at proxy size: + on the main subject (centre-weighted salient area and the people),
    - towards the edges (burn), never on faces or the subject."""
    lt = {**LIGHT_DEFAULTS, **light}
    ph, pw = p.shape[:2]
    long = max(ph, pw)
    stops = np.zeros((ph, pw), np.float32)
    subj = np.zeros((ph, pw), np.float32)
    faces = people_mask(subject, shape)
    if faces is not None and faces.shape != (ph, pw):
        faces = cv2.resize(faces, (pw, ph), interpolation=cv2.INTER_LINEAR)
    if lt["subject"]:
        # Faces are never lifted here: person exposure is the face's tool, and sunlit skin clips red first.
        subj = _blur(_food(p), LIGHT_BLUR * long)
        subj /= max(float(subj.max()), 1e-6)
        if faces is not None:
            subj *= 1 - faces
        stops += SUBJECT_STOPS * lt["subject"] * subj
    if lt["burn"]:
        yy = (np.arange(ph, dtype=np.float32) + 0.5) / ph * 2 - 1
        xx = (np.arange(pw, dtype=np.float32) + 0.5) / pw * 2 - 1
        d = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2) / math.sqrt(2)
        e = _smoothstep(BURN_START, 1.0, d).astype(np.float32)
        e *= 1 - subj
        if faces is not None:
            e *= 1 - faces
        stops -= BURN_STOPS * lt["burn"] * e
    return stops


def local_light(x: np.ndarray, light: dict, subject: Subject | None) -> None:
    """In place: the subject is lifted (easing off towards white, so nothing clips) and the edges burnt."""
    lt = {**LIGHT_DEFAULTS, **(light or {})}
    if not (lt["subject"] or lt["burn"]):
        return
    full = upsample(light_stops(x.shape, lt, subject, _proxy(x)), x.shape)
    for r in range(0, x.shape[0], STRIP):
        xs, g = x[r:r + STRIP], full[r:r + STRIP]
        lift = (2 ** (np.maximum(g, 0) / 2.2) - 1) * (1 - _smoothstep(*LIFT_KNEE, xs.max(-1)))
        xs *= ((1 + lift) * 2 ** (np.minimum(g, 0) / 2.2))[..., None]
