"""Look and finish parameters, their allowed ranges and validation, and the strength presets."""

LOOK_DEFAULTS = {
    "contrast": 0.10, "vibrance": 0.20, "warmth": 0.025, "skin_protect": 0.7,
    "shadows": 0.0, "highlights": 0.0, "black": 0.0, "saturation": 1.0,
    "orange_sat": 1.0, "blue_sat": 1.0, "blue_lum": 0.0, "tint": 0.0,
}

LOOK_RANGES = {
    "contrast": (-0.5, 0.5), "vibrance": (-1.0, 1.0), "warmth": (-0.15, 0.15), "skin_protect": (0.0, 1.0),
    "shadows": (0.0, 0.15), "highlights": (0.0, 0.4), "black": (0.0, 0.05), "saturation": (0.7, 1.3),
    "orange_sat": (0.6, 1.2), "blue_sat": (0.7, 1.3), "blue_lum": (-0.1, 0.1), "tint": (-0.03, 0.03),
}


def look_errors(look: dict) -> list[str]:
    errs = []
    for k, v in look.items():
        if k not in LOOK_RANGES:
            errs.append(f"unknown look parameter {k!r} (allowed: {', '.join(LOOK_RANGES)})")
            continue
        lo, hi = LOOK_RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            errs.append(f"look parameter {k}={v!r} must be a number in {lo}..{hi}")
    return errs


# Finish. Defaults are no-ops, so an empty finish leaves a photo untouched.
FINISH_DEFAULTS = {"exposure": 0.0, "wb_r": 1.0, "wb_b": 1.0, "straighten": 0.0, "denoise": 0.0, "clarity": 0.0,
                   "sharpen": 0.0, "vignette": 0.0, "grain": 0.0, "dehaze": 0.0, "texture": 0.0}

FINISH_RANGES = {
    "exposure": (-1.5, 1.5), "wb_r": (0.9, 1.1), "wb_b": (0.9, 1.1), "straighten": (-10.0, 10.0),
    "denoise": (0.0, 1.0), "clarity": (0.0, 1.0), "sharpen": (0.0, 1.5), "vignette": (0.0, 1.0), "grain": (0.0, 1.0),
    "dehaze": (0.0, 1.0), "texture": (0.0, 1.0),
}


# finish "person": local enhancement of the people in a photo (face lift, gentle local contrast, natural skin,
# sharp eyes). Defaults are no-ops; skin_smooth tops out at 0.5 so skin can never go plastic.
PERSON_DEFAULTS = {"exposure": 0.0, "contrast": 0.0, "clarity": 0.0, "skin_smooth": 0.0, "eyes": 0.0}
PERSON_RANGES = {"exposure": (-1.0, 1.0), "contrast": (-0.3, 0.3), "clarity": (0.0, 0.5), "skin_smooth": (0.0, 0.5),
                 "eyes": (0.0, 1.0)}


def _range_errors(values: dict, ranges: dict, what: str) -> list[str]:
    errs = []
    for k, v in values.items():
        if k not in ranges:
            errs.append(f"unknown {what} parameter {k!r} (allowed: {', '.join(ranges)})")
            continue
        lo, hi = ranges[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            errs.append(f"{what} parameter {k}={v!r} must be a number in {lo}..{hi}")
    return errs


def person_errors(person) -> list[str]:
    if not isinstance(person, dict):
        return ["finish parameter person must be a dictionary of person settings "
                f"({', '.join(PERSON_RANGES)})"]
    return _range_errors(person, PERSON_RANGES, "person")


def finish_errors(finish: dict) -> list[str]:
    errs = person_errors(finish["person"]) if "person" in finish else []
    if "light" in finish:
        errs += _dict_errors(finish["light"], LIGHT_RANGES, "light")
    for k, v in finish.items():
        if k in ("person", "light"):
            continue
        if k not in FINISH_RANGES:
            errs.append(f"unknown finish parameter {k!r} (allowed: {', '.join(FINISH_RANGES)}, person, light)")
            continue
        lo, hi = FINISH_RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            errs.append(f"finish parameter {k}={v!r} must be a number in {lo}..{hi}")
    return errs


# ---- develop ops ----
# finish "light": local light. subject brightens the main subject (salient area and the people), burn darkens
# the edges (softer and larger than vignette, never on faces). Both 0..0.5, defaults no-ops.
LIGHT_DEFAULTS = {"subject": 0.0, "burn": 0.0}
LIGHT_RANGES = {"subject": (0.0, 0.5), "burn": (0.0, 0.5)}

# Look ops that hold dictionaries, kept out of LOOK_DEFAULTS (look_transform takes only plain numbers).
CURVE_DEFAULTS = {"strength": 0.0, "black": 0.0, "white": 1.0, "pivot": 0.5}
CURVE_RANGES = {"strength": (0.0, 0.6), "black": (0.0, 0.08), "white": (0.92, 1.0), "pivot": (0.35, 0.65)}
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_BAND_DEFAULTS = {"h": 0.0, "s": 0.0, "l": 0.0}
HSL_BAND_RANGES = {"h": (-15.0, 15.0), "s": (-0.5, 0.5), "l": (-0.3, 0.3)}
HSL_DEFAULTS = {b: dict(HSL_BAND_DEFAULTS) for b in HSL_BANDS}
WHEELS = ("shadows", "midtones", "highlights")
WHEEL_DEFAULTS = {"hue": 0.0, "sat": 0.0}
WHEEL_RANGES = {"hue": (0.0, 360.0), "sat": (0.0, 0.25)}
GRADE_DEFAULTS = {**{w: dict(WHEEL_DEFAULTS) for w in WHEELS}, "balance": 0.0, "blend": 0.5}
GRADE_RANGES = {"balance": (-1.0, 1.0), "blend": (0.0, 1.0)}
LOOK_OPS = {"curve": CURVE_DEFAULTS, "hsl": HSL_DEFAULTS, "grade": GRADE_DEFAULTS}


def _dict_errors(values, ranges: dict, what: str) -> list[str]:
    if not isinstance(values, dict):
        return [f"{what} must be a dictionary ({', '.join(ranges)})"]
    return _range_errors(values, ranges, what)


def look_op_errors(name: str, value) -> list[str]:
    if name == "curve":
        return _dict_errors(value, CURVE_RANGES, "curve")
    if not isinstance(value, dict):
        what = ", ".join(HSL_BANDS) if name == "hsl" else ", ".join((*WHEELS, *GRADE_RANGES))
        return [f"{name} must be a dictionary ({what})"]
    errs = []
    for k, v in value.items():
        if name == "hsl":
            if k not in HSL_BANDS:
                errs.append(f"unknown hsl band {k!r} (allowed: {', '.join(HSL_BANDS)})")
            else:
                errs += _dict_errors(v, HSL_BAND_RANGES, f"hsl {k}")
        elif k in WHEELS:
            errs += _dict_errors(v, WHEEL_RANGES, f"grade {k}")
        elif k in GRADE_RANGES:
            errs += _range_errors({k: v}, GRADE_RANGES, "grade")
        else:
            errs.append(f"unknown grade parameter {k!r} (allowed: {', '.join((*WHEELS, *GRADE_RANGES))})")
    return errs


def full_look_errors(look: dict) -> list[str]:
    """look_errors plus the look ops curve, hsl and grade."""
    errs = look_errors({k: v for k, v in look.items() if k not in LOOK_OPS})
    for k in LOOK_OPS:
        if k in look:
            errs += look_op_errors(k, look[k])
    return errs


def merge(base: dict, *over: dict) -> dict:
    """Key-by-key merge; dictionaries (curve, hsl bands, grade wheels, light) merge one level deeper too."""
    out = dict(base)
    for o in over:
        for k, v in o.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def look_with_ops(look: dict) -> dict:
    """The look with every look op filled in from its defaults."""
    return merge({k: merge(v) for k, v in LOOK_OPS.items()}, look)


def split_look(look: dict) -> tuple[dict, dict]:
    """(the look_transform keys, the look ops) of a look."""
    return {k: v for k, v in look.items() if k not in LOOK_OPS}, {k: look[k] for k in LOOK_OPS if k in look}


# Strength presets: complete starting grades (base look, develop ops and a finish). Settings you give override a
# preset key by key. A preset's dehaze is scaled by each photo's measured haze, so on clear air the punch comes from
# curve, HSL, wheels and light. Sharpen, texture and clarity stay low because on phone night shots and action-camera
# stills they read as grain; there is no denoise because it paints watercolour skies.
PRESETS = {
    "natural": {
        "look": {"contrast": 0.14, "vibrance": 0.26, "warmth": 0.01, "skin_protect": 0.85, "shadows": 0.03,
                 "highlights": 0.2, "orange_sat": 0.88, "blue_sat": 0.97, "blue_lum": -0.03,
                 "curve": {"strength": 0.1, "pivot": 0.45},
                 "hsl": {"orange": {"s": 0.04}, "blue": {"l": -0.03}},
                 "grade": {"highlights": {"hue": 40, "sat": 0.03}, "shadows": {"hue": 200, "sat": 0.02}}},
        "finish": {"sharpen": 0.15, "clarity": 0.1, "dehaze": 0.1, "texture": 0.05,
                   "light": {"subject": 0.08, "burn": 0.08}},
    },
    "punchy": {
        "look": {"contrast": 0.18, "vibrance": 0.3, "warmth": 0.01, "skin_protect": 0.85, "shadows": 0.02,
                 "highlights": 0.24, "orange_sat": 0.85, "blue_sat": 1.0, "blue_lum": -0.03,
                 "curve": {"strength": 0.45, "black": 0.025, "pivot": 0.45},
                 "hsl": {"red": {"s": 0.12}, "orange": {"s": 0.18}, "yellow": {"s": 0.12},
                         "aqua": {"l": -0.06}, "blue": {"s": 0.1, "l": -0.14}},
                 "grade": {"highlights": {"hue": 40, "sat": 0.11}, "shadows": {"hue": 205, "sat": 0.08}}},
        "finish": {"sharpen": 0.2, "clarity": 0.15, "dehaze": 0.3, "texture": 0.1,
                   "light": {"subject": 0.3, "burn": 0.3}},
    },
    "bold": {
        "look": {"contrast": 0.22, "vibrance": 0.32, "warmth": 0.01, "skin_protect": 0.85, "shadows": 0.02,
                 "highlights": 0.26, "orange_sat": 0.82, "blue_sat": 1.0, "blue_lum": -0.03,
                 "curve": {"strength": 0.6, "black": 0.035, "pivot": 0.45},
                 "hsl": {"red": {"s": 0.16}, "orange": {"s": 0.22}, "yellow": {"s": 0.16},
                         "aqua": {"l": -0.1}, "blue": {"s": 0.14, "l": -0.2}},
                 "grade": {"highlights": {"hue": 40, "sat": 0.15}, "shadows": {"hue": 205, "sat": 0.11}}},
        "finish": {"sharpen": 0.25, "clarity": 0.2, "dehaze": 0.45, "texture": 0.15,
                   "light": {"subject": 0.4, "burn": 0.4}},
    },
}


def preset_errors(preset) -> list[str]:
    if preset is None or (isinstance(preset, str) and preset in PRESETS):
        return []
    return [f"preset {preset!r} isn't one of {', '.join(PRESETS)}"]
