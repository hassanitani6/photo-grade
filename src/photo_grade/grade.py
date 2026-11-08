"""The grading engine: a look (tone and colour) plus finish ops (straighten, denoise, exposure, white balance,
dehaze, clarity, texture, sharpen, local light, vignette, grain), scene matching and 16-bit TIFF output."""
import hashlib
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .looks import FINISH_DEFAULTS, LOOK_DEFAULTS, split_look

LUMA = np.array([0.2126, 0.7152, 0.0722])
NEUTRAL = {"exposure": 1.0, "r": 1.0, "b": 1.0}
SKIN_HUE, SKIN_WIDTH = 25.0, 20.0


def _hue_deg(x: np.ndarray) -> np.ndarray:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360


def _smoothstep(lo: float, hi: float, v: np.ndarray) -> np.ndarray:
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _hue_band(hue: np.ndarray, centre: float, half: float, feather: float) -> np.ndarray:
    d = np.abs(((hue - centre + 180) % 360) - 180)
    return np.clip((half + feather - d) / feather, 0, 1)


def look_transform(rgb: np.ndarray, contrast: float, vibrance: float, warmth: float, skin_protect: float,
                   shadows: float = 0.0, highlights: float = 0.0, black: float = 0.0, saturation: float = 1.0,
                   orange_sat: float = 1.0, blue_sat: float = 1.0, blue_lum: float = 0.0,
                   tint: float = 0.0) -> np.ndarray:
    x = np.clip(rgb, 0, 1)
    x = x + contrast * (x * x * (3 - 2 * x) - x)
    if shadows or highlights:
        luma = (x @ LUMA)[..., None]
        if shadows:
            # Bell over the low-mids: zero at luma 0 (black point stays put) and at 0.6 and above.
            x = x + shadows * _smoothstep(0, 0.08, luma) * (1 - _smoothstep(0, 0.6, luma))
        if highlights:
            # Pulls 0.7..1 down with the peak near 0.9 and nothing at 1.0, so whites stay white and the curve stays monotonic.
            u = np.clip((luma - 0.7) / 0.3, 0, 1)
            x = x - highlights * 2 * u * u * (1 - u)
    if black:
        x = black + x * (1 - black)
    luma = (x @ LUMA)[..., None]
    sat = x.max(-1, keepdims=True) - x.min(-1, keepdims=True)
    hue = _hue_deg(x)
    hue_dist = np.abs(((hue - SKIN_HUE + 180) % 360) - 180)
    skin = np.clip(1 - hue_dist / SKIN_WIDTH, 0, 1)[..., None] * np.clip(sat * 4, 0, 1)
    # Vibrance boosts muted colours more than saturated ones; skin-hued pixels get less of it.
    gain = 1 + vibrance * (1 - sat) * (1 - skin_protect * skin)
    x = luma + (x - luma) * gain
    if saturation != 1.0:
        x = luma + (x - luma) * (1 + (saturation - 1) * (1 - skin_protect * skin))
    chroma = np.clip(sat * 4, 0, 1)
    if orange_sat != 1.0:
        # Sand is skin-coloured but far more saturated, so skin protection fades out above sat 0.4 (fully gone at 0.6).
        sand = _hue_band(hue, 37.5, 17.5, 10)[..., None] * chroma
        guard = 1 - skin_protect * skin * (1 - np.clip((sat - 0.4) / 0.2, 0, 1))
        x = luma + (x - luma) * (1 + (orange_sat - 1) * sand * guard)
    if blue_sat != 1.0 or blue_lum:
        sea = _hue_band(hue, 215, 25, 15)[..., None] * chroma
        if blue_sat != 1.0:
            x = luma + (x - luma) * (1 + (blue_sat - 1) * sea)
        if blue_lum:
            x = x + blue_lum * sea
    highlight_mask = np.clip((luma - 0.5) * 2, 0, 1)
    x = x + warmth * highlight_mask * np.array([1.0, 0.35, -0.8])
    if tint:
        # Positive = magenta, negative = green; strongest at mid grey.
        x = x + tint * 4 * luma * (1 - luma) * np.array([0.5, -1.0, 0.5])
    return np.clip(x, 0, 1)



def _midtone_mean(a: np.ndarray) -> np.ndarray:
    flat = a.reshape(-1, 3)
    lum = flat @ LUMA
    mask = (lum > 0.15) & (lum < 0.85)
    return flat[mask].mean(0) if mask.sum() > 20 else flat.mean(0)


def match_params(ref_rgb: np.ndarray, shot_rgb: np.ndarray, max_stop: float = 0.5, max_wb: float = 0.08) -> dict:
    r, s = _midtone_mean(ref_rgb), _midtone_mean(shot_rgb)
    eps = 1e-4
    exposure = np.clip((r @ LUMA + eps) / (s @ LUMA + eps), 2 ** -max_stop, 2 ** max_stop)
    rg = np.clip(((r[0] + eps) / (r[1] + eps)) / ((s[0] + eps) / (s[1] + eps)), 1 - max_wb, 1 + max_wb)
    bg = np.clip(((r[2] + eps) / (r[1] + eps)) / ((s[2] + eps) / (s[1] + eps)), 1 - max_wb, 1 + max_wb)
    return {"exposure": float(exposure), "r": float(rg), "b": float(bg)}

# Before/after previews and haze estimates use a copy this long on its long side, downscaled before grading for
# speed: region medians and midtone means barely change with scale; only per-pixel texture (sharpen, grain) does.
PREVIEW_SIDE = 540
STRIP = 256  # rows per chunk for per-pixel ops: keeps float64 temporaries at ~35 MB on a 5712-px photo
_LUMA32 = LUMA.astype(np.float32)
# Vignette angle per unit (0.6 = PI/7), as in ffmpeg's vignette filter.
VIGNETTE_PER_UNIT = (math.pi / 7) / 0.6
# grain 1.0 = noise of 3/255. It's drawn on a grid of at most GRAIN_GRID cells along the long side and enlarged,
# so it survives a later downscale (e.g. to a 1080-px social post) instead of averaging away.
GRAIN_SIGMA = 3.0 / 255
GRAIN_GRID = 1600
GRAIN_SEED = 7919  # used when apply isn't told which photo it's grading


class GradeError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def straighten_geometry(w: int, h: int, deg: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """(2x3 affine from original px to the rotated w x h canvas, crop box (x0, y0, cw, ch) on that canvas).

    Positive deg turns the picture anticlockwise (levels a horizon that runs downhill to the right). The crop is
    the largest centred rectangle of the same aspect inside the rotated frame, 2 px in so interpolation at the
    rotated edge never reaches it."""
    if not deg:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (0, 0, w, h)
    t = math.radians(abs(deg))
    c, s = math.cos(t), math.sin(t)
    k = min(w / (w * c + h * s), h / (w * s + h * c))
    cw = max(1, math.floor(w * k) - 2)
    ch = max(1, min(round(cw * h / w), math.floor(h * k) - 2))
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), deg, 1.0)
    return m, ((w - cw) // 2, (h - ch) // 2, cw, ch)


def graded_size(w: int, h: int, deg: float) -> tuple[int, int]:
    """(width, height) of the graded file for a w x h original straightened by deg."""
    _, (_, _, cw, ch) = straighten_geometry(w, h, deg)
    return cw, ch


def map_box(box: tuple[int, int, int, int], w: int, h: int, deg: float) -> tuple[int, int, int, int] | None:
    """An (x0, y0, x1, y1) box in original px -> graded-file px: bounding box of its rotated corners, clipped
    to the crop. None when nothing of it survives the crop."""
    m, (cx, cy, cw, ch) = straighten_geometry(w, h, deg)
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], float) @ m.T
    bx0, by0 = max(0, math.floor(pts[:, 0].min() - cx + 1e-6)), max(0, math.floor(pts[:, 1].min() - cy + 1e-6))
    bx1, by1 = min(cw, math.ceil(pts[:, 0].max() - cx - 1e-6)), min(ch, math.ceil(pts[:, 1].max() - cy - 1e-6))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return bx0, by0, bx1, by1


def map_points(points, w: int, h: int, deg: float) -> list[tuple[float, float]]:
    """(x, y) points in original px -> graded-file px (rotated, then moved into the crop)."""
    m, (cx, cy, _, _) = straighten_geometry(w, h, deg)
    pts = np.array([[x, y, 1] for x, y in points], float) @ m.T
    return [(float(a - cx), float(b - cy)) for a, b in pts]


def _straighten(x: np.ndarray, deg: float) -> np.ndarray:
    h, w = x.shape[:2]
    m, (cx, cy, cw, ch) = straighten_geometry(w, h, deg)
    m = m.copy()
    # Shifted so the warp writes only the crop: no full-size rotated intermediate.
    m[0, 2] -= cx
    m[1, 2] -= cy
    return cv2.warpAffine(x, m, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _denoise(x: np.ndarray, amount: float) -> None:
    h = 3 + 7 * amount
    src = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
    # OpenCV's colour denoiser works in Lab converted from BGR, so it must be given BGR.
    den = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(src[..., ::-1]), None, h, h, 7, 21)[..., ::-1]
    for r in range(0, x.shape[0], STRIP):
        x[r:r + STRIP] += (den[r:r + STRIP].astype(np.float32) - src[r:r + STRIP]) / 255


def _gains(finish: dict, match: dict | None) -> np.ndarray:
    # Divided by 2.2 because the gain acts on gamma-encoded values: approximate stops.
    e = 2 ** (finish["exposure"] / 2.2)
    m = match or NEUTRAL
    me = e * m["exposure"]
    return np.array([me * m["r"] * finish["wb_r"], me, me * m["b"] * finish["wb_b"]], np.float32)


def _luma_unsharp(x: np.ndarray, sigma: float, amount: float) -> None:
    y = x @ _LUMA32
    y -= cv2.GaussianBlur(y, (0, 0), sigma)
    y *= amount
    x += y[..., None]


def _vignette(x: np.ndarray, v: float) -> None:
    # ffmpeg-style vignette: gain cos(angle * d)^4, d = distance from centre / centre-to-corner.
    h, w = x.shape[:2]
    angle = v * VIGNETTE_PER_UNIT
    dmax = math.hypot((w - 1) / 2, (h - 1) / 2) or 1.0
    xx = ((np.arange(w, dtype=np.float32) - (w - 1) / 2) / dmax) ** 2
    yy = ((np.arange(h, dtype=np.float32) - (h - 1) / 2) / dmax) ** 2
    for r in range(0, h, STRIP):
        f = np.cos(angle * np.sqrt(yy[r:r + STRIP, None] + xx[None, :])) ** 4
        x[r:r + STRIP] *= f[..., None]


def _grain(x: np.ndarray, g: float, seed: int) -> None:
    """Monochrome, seeded per photo, so the same photo always gets the same grain."""
    h, w = x.shape[:2]
    k = max(1.0, max(h, w) / GRAIN_GRID)
    gh, gw = max(1, round(h / k)), max(1, round(w / k))
    n = np.random.default_rng(seed).standard_normal((gh, gw), dtype=np.float32)
    if (gh, gw) != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        n /= max(float(n.std()), 1e-6)
    n *= g * GRAIN_SIGMA
    x += n[..., None]


def grain_seed(name: str) -> int:
    """A stable grain seed per photo name, so re-grading a photo gives the same grain."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def apply(rgb: np.ndarray, look: dict, finish: dict, match: dict | None, *, copy: bool = True,
          scale: float = 1.0, seed: int | None = None, regions: dict | None = None, subject=None,
          final_skin_guard: bool = True) -> np.ndarray:
    """rgb: HxWx3 floats 0..1, sRGB as displayed. Returns float32 0..1 (smaller than the input when straightened).

    Order: straighten, denoise, exposure, match + white balance, dehaze, clarity, texture, sharpen, look, curve,
    HSL, colour wheels, region looks, person (finish "person"), local light, vignette, grain. Region masks are read
    from the image just before the look.
    copy=False lets the caller hand over a float32 array it no longer needs, saving a full-size copy.
    scale = this image's long side / the original's (previews); seed = grain_seed(photo id).
    regions: {region: look} (regions.REGION_NAMES); subject: regions.Subject with the faces and category, in
    the straightened photo's px (rescaled to this image). final_skin_guard: after everything but vignette and
    grain, ease back red the grade pushed towards clipping on the guarded faces (regions.guard_final_skin)."""
    from . import develop
    # Imported here: regions imports grade (look_transform) and measure, which imports grade.
    from . import regions as regionsmod
    look, ops = split_look({**LOOK_DEFAULTS, **look})
    ops = {**ops, "skin_protect": look["skin_protect"]}
    f = {**FINISH_DEFAULTS, **finish}
    x = np.array(rgb, dtype=np.float32, order="C") if copy else np.ascontiguousarray(rgb, dtype=np.float32)
    if f["straighten"]:
        x = _straighten(x, f["straighten"])
    if f["denoise"]:
        _denoise(x, f["denoise"])
    g = _gains(f, match)
    if not np.all(g == 1):
        x *= g
    final = regionsmod.final_skin_boxes(subject, x.shape) if final_skin_guard else []
    source = [x[b[1]:b[3], b[0]:b[2]].copy() for b in final]
    if f["dehaze"]:
        develop.dehaze(x, f["dehaze"], subject)
    if f["clarity"]:
        develop.clarity(x, f["clarity"], subject)
    if f["texture"]:
        develop.texture(x, f["texture"], subject)
    if f["sharpen"]:
        # A 1-px sharpen on the full photo is mostly averaged away when it's shrunk to a preview, so the
        # preview's radius shrinks with it (floor 0.3 px) to show about what the downscaled render shows.
        _luma_unsharp(x, max(0.3, 1.0 * scale), f["sharpen"])
    person = f.get("person")
    if regions or person:
        active = regionsmod.active_regions(regions)
        subj = subject or regionsmod.Subject((x.shape[1], x.shape[0]))
        masks = regionsmod.region_masks(x, subj, tuple(active)) if active else {}
    # look_transform is per-pixel, so running it in row strips is exact; it promotes to float64 (LUMA is
    # float64), and strips keep that from doubling a ~290 MB full-size image. Assignment casts back to float32.
    guarded = regionsmod.warmth_guard_boxes(subject, x.shape, look) if subject is not None else []
    saved = [x[b[1]:b[3], b[0]:b[2]].copy() for b in guarded]
    for r in range(0, x.shape[0], STRIP):
        x[r:r + STRIP] = look_transform(x[r:r + STRIP], **look)
    if guarded:
        regionsmod.guard_skin_warmth(x, subject, look, saved, guarded)
    people = None
    if develop.ops_active(ops) or (regions and any(develop.ops_active(v) for v in active.values())):
        people = develop.people_mask(subject, x.shape)
        people = regionsmod.upsample(people, x.shape) if people is not None else None
    if develop.ops_active(ops):
        for r in range(0, x.shape[0], STRIP):
            x[r:r + STRIP] = develop.look_ops(x[r:r + STRIP], ops, None if people is None else people[r:r + STRIP])
    if regions or person:
        if masks:
            regionsmod.apply_regions(x, masks, active, people)
            del masks
        regionsmod.enhance_person(x, subject, person)
    del people
    if f.get("light"):
        develop.local_light(x, f["light"], subject)
    if final:
        regionsmod.guard_final_skin(x, subject, look["skin_protect"], source, final)
    if f["vignette"]:
        _vignette(x, f["vignette"])
    if f["grain"]:
        _grain(x, f["grain"], GRAIN_SEED if seed is None else seed)
    return np.clip(x, 0, 1, out=x)


def write_graded(rgb: np.ndarray, dest: Path) -> Path:
    """16-bit RGB TIFF via OpenCV (Pillow can't write 16-bit RGB TIFF); .part then rename."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    u16 = np.empty(rgb.shape, np.uint16)
    for r in range(0, rgb.shape[0], STRIP):
        u16[r:r + STRIP] = np.rint(rgb[r:r + STRIP, :, ::-1] * 65535)
    # Uncompressed: on a 24 MP photo deflate saved under 10% (photo noise fills the low bits) for ~6 s more.
    # Written straight to a .part.tif (imwrite picks the encoder from the extension), then renamed.
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.part.tif")
    try:
        if not cv2.imwrite(str(tmp), u16, [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_NONE]):
            raise GradeError([f"Couldn't write the graded file {dest.name}."])
        del u16
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def read_graded(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.ndim != 3:
        raise GradeError([f"The graded file {Path(path).name} couldn't be read."])
    out = bgr[..., ::-1].astype(np.float32)
    out /= 65535
    return out


def fit(arr: np.ndarray, box: int) -> Image.Image:
    """A float RGB array as an 8-bit image no larger than box on its long side."""
    im = Image.fromarray(np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8), "RGB")
    if max(im.size) > box:
        k = box / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im
