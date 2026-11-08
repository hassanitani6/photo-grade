"""photo-grade: colour-grade photos from the command line.

    photo-grade PHOTO_OR_FOLDER... [--preset punchy] [-o graded/]
    photo-grade presets            show what each preset does
    photo-grade doctor             environment report for bug reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import __version__

PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
SETTINGS_KEYS = {"preset", "look", "finish", "regions", "label"}
LABELS = ("food", "sunset")  # subject labels the engine uses: the food region and the warm sunset sky


@dataclass
class Settings:
    look: dict
    finish: dict
    regions: dict
    preset: str | None
    own_finish: dict  # the finish keys the user set explicitly (these are never rescaled)
    label: str | None = None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="photo-grade", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"photo-grade {__version__}")
    p.add_argument("inputs", nargs="*", type=Path, help="photos or folders of photos")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("graded"), help="output folder (default: ./graded)")
    p.add_argument("--preset", choices=("natural", "punchy", "bold", "none"), default="natural",
                   help="starting grade (default: natural)")
    p.add_argument("--settings", type=Path, help="JSON file with preset/look/finish/regions/label overrides")
    p.add_argument("--format", choices=("jpeg", "tiff16"), default="jpeg", help="output format (default: jpeg)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    p.add_argument("--suffix", default="_graded", help="added to each output file name (default: _graded)")
    p.add_argument("--before-after", action="store_true", help="also write a side-by-side before/after preview")
    p.add_argument("--recursive", action="store_true", help="look for photos in sub-folders too")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def load_settings(path: Path | None, preset_flag: str) -> Settings:
    """The preset merged with the settings file, validated. Raises ValueError listing every problem."""
    from .looks import PRESETS, finish_errors, full_look_errors, merge, preset_errors
    from .regions import region_errors
    raw: dict = {}
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"couldn't read settings file {path.name}: {e}") from None
        if not isinstance(raw, dict):
            raise ValueError("the settings file must be a JSON object with preset, look, finish and/or regions")
    problems = [f"unknown section {k!r} (allowed: {', '.join(sorted(SETTINGS_KEYS))})" for k in raw if k not in SETTINGS_KEYS]
    preset = raw.get("preset", preset_flag)
    preset = None if preset == "none" else preset
    problems += preset_errors(preset)
    label = raw.get("label")
    if label is not None and label not in LABELS:
        problems.append(f"label {label!r} isn't one of {', '.join(LABELS)}")
    look, finish, regions = raw.get("look", {}), raw.get("finish", {}), raw.get("regions", {})
    for name, val in (("look", look), ("finish", finish), ("regions", regions)):
        if not isinstance(val, dict):
            problems.append(f"{name} must be a JSON object")
    if not problems:
        problems += full_look_errors(look) + finish_errors(finish) + region_errors(regions)
    if problems:
        raise ValueError("; ".join(problems))
    base = PRESETS[preset] if preset else {"look": {}, "finish": {}}
    return Settings(merge(base["look"], look), merge(base["finish"], finish), regions, preset, finish, label)


def _within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def find_photos(inputs: list[Path], recursive: bool, skip_dir: Path | None = None) -> list[Path]:
    """Photos from files and folders, in order, each real file once. Files inside skip_dir (the output folder)
    are never picked up from a folder, so a second run doesn't grade its own outputs."""
    found, seen = [], set()
    for p in inputs:
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            cands = sorted(f for f in it if f.is_file() and f.suffix.lower() in PHOTO_TYPES
                           and not f.name.startswith(".") and not (skip_dir and _within(f, skip_dir)))
        else:
            cands = [p]
        for f in cands:
            try:
                st = f.stat()
                key = (st.st_dev, st.st_ino)
            except OSError:
                key = ("missing", str(f))
            if key not in seen:
                seen.add(key)
                found.append(f)
    return found


def output_path(src: Path, out_dir: Path, suffix: str, fmt: str) -> Path:
    return out_dir / f"{src.stem}{suffix}{'.jpg' if fmt == 'jpeg' else '.tif'}"


def sheet_path(src: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{src.stem}{suffix}_before_after.jpg"


def planned_outputs(src: Path, args) -> list[Path]:
    outs = [output_path(src, args.out_dir, args.suffix, args.format)]
    if args.before_after:
        outs.append(sheet_path(src, args.out_dir, args.suffix))
    return outs


def output_clashes(photos: list[Path], args) -> list[str]:
    """Pairs of inputs that would write the same output file (compared case-insensitively, because macOS and
    Windows treat names that differ only in case as one file)."""
    owner: dict[str, Path] = {}
    clashes = []
    for src in photos:
        for out in planned_outputs(src, args):
            key = str(out.resolve()).casefold()
            if key in owner and owner[key] != src:
                clashes.append(f"{owner[key]} and {src} would both write {out.name}")
            owner.setdefault(key, src)
    return clashes


def _check_destination(dest: Path, src: Path, overwrite: bool) -> None:
    if dest.exists():
        # samefile catches every way two names can be one file: identical paths, a different letter case on
        # case-insensitive disks, symbolic links and hard links.
        if dest.resolve() == src.resolve() or src.exists() and os.path.samefile(dest, src):
            raise ValueError("the output would replace the original; choose another --out-dir or --suffix")
        if not overwrite:
            raise ValueError(f"{dest} already exists (use --overwrite to replace it)")


def grade_one(src: Path, s: Settings, args) -> list[Path]:
    from PIL import Image

    from .develop import haze_factor
    from .grade import PREVIEW_SIDE, apply, fit, grain_seed, write_graded
    from .imageio import load_rgb, save_image_jpeg, save_jpeg
    from .regions import Subject

    outs = planned_outputs(src, args)
    for dest in outs:
        _check_destination(dest, src, args.overwrite)
    finish = dict(s.finish)
    scale_dehaze = bool(s.preset and finish.get("dehaze") and "dehaze" not in s.own_finish)
    preview = load_rgb(src, max_side=PREVIEW_SIDE) if (scale_dehaze or args.before_after) else None
    if scale_dehaze:
        # A preset's dehaze follows the photo's measured haze, so clear and indoor photos get almost none.
        finish["dehaze"] = round(finish["dehaze"] * haze_factor(preview), 3)
    seed = grain_seed(src.name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(src)
    long_side = max(rgb.shape[:2])
    subject = Subject((rgb.shape[1], rgb.shape[0]), label=s.label) if s.label else None
    graded = apply(rgb, s.look, finish, None, copy=False, seed=seed, regions=s.regions or None, subject=subject)
    del rgb
    written = [save_jpeg(graded, outs[0], quality=args.quality) if args.format == "jpeg" else write_graded(graded, outs[0])]
    del graded
    if args.before_after:
        after = apply(preview, s.look, finish, None, scale=max(preview.shape[:2]) / long_side, seed=seed,
                      regions=s.regions or None, subject=subject)
        before_im, after_im = fit(preview, PREVIEW_SIDE), fit(after, PREVIEW_SIDE)
        sheet = Image.new("RGB", (before_im.width + after_im.width + 30, max(before_im.height, after_im.height) + 20),
                          (24, 24, 24))
        sheet.paste(before_im, (10, 10))
        sheet.paste(after_im, (20 + before_im.width, 10))
        written.append(save_image_jpeg(sheet, outs[1], quality=90))
    return written


def _grade(args) -> int:
    from .grade import GradeError
    from .imageio import READ_ERRORS, read_error_why
    if not args.inputs:
        print("error: give at least one photo or folder (see --help)", file=sys.stderr)
        return 2
    if not 1 <= args.quality <= 100:
        print("error: --quality must be between 1 and 100", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.settings, args.preset)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if any((settings.finish.get("person") or {}).values()):
        print("note: 'person' settings need face positions, which photo-grade does not detect; they are ignored.",
              file=sys.stderr)
    photos = find_photos(args.inputs, args.recursive, skip_dir=args.out_dir)
    if not photos:
        print("error: no photos found (looked for " + ", ".join(sorted(PHOTO_TYPES)) + ")", file=sys.stderr)
        return 1
    clashes = output_clashes(photos, args)
    if clashes:
        print("error: these photos would write the same output file, so nothing was graded:\n  " + "\n  ".join(clashes)
              + "\nGrade them in separate runs or with a different --out-dir.", file=sys.stderr)
        return 2
    failed = 0
    for i, src in enumerate(photos, 1):
        try:
            if not src.is_file():
                raise ValueError("file not found")
            for out in grade_one(src, settings, args):
                if not args.quiet:
                    print(f"[{i}/{len(photos)}] {src.name} -> {out}")
        except (ValueError, GradeError, *READ_ERRORS) as e:
            failed += 1
            why = str(e) if isinstance(e, (ValueError, GradeError)) and not isinstance(e, READ_ERRORS) else read_error_why(e)
            print(f"[{i}/{len(photos)}] {src.name}: skipped ({why})", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(photos)} photo(s) were not graded.", file=sys.stderr)
    return 1 if failed else 0


def _presets() -> int:
    from .looks import PRESETS
    print(json.dumps(PRESETS, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    argv = list(sys.argv[1:] if argv is None else argv)
    allowed = {"doctor": {"--json"}, "presets": set()}
    if argv[:1] and argv[0] in allowed:
        extra = [a for a in argv[1:] if a not in allowed[argv[0]]]
        if extra:
            print(f"error: unexpected argument(s) for {argv[0]}: {' '.join(extra)} "
                  f"(to grade a file with that name, write ./{argv[0]})", file=sys.stderr)
            return 2
        if argv[0] == "doctor":
            from .doctor import run
            return run(as_json="--json" in argv[1:])
        return _presets()
    return _grade(_parser().parse_args(argv))
