import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from photo_grade import doctor
from photo_grade.cli import load_settings, main

from conftest import make_heic, make_jpeg  # noqa: E402


def scene(path, size=(240, 160)):
    """A synthetic sky-over-sand photo."""
    w, h = size
    arr = np.zeros((h, w, 3), np.uint8)
    arr[: h // 2] = (140, 175, 225)
    arr[h // 2:] = (200, 165, 115)
    Image.fromarray(arr).save(path)
    return path


def run(*args, cwd):
    return subprocess.run([sys.executable, "-m", "photo_grade", *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


def test_grades_a_photo_into_the_output_folder(tmp_path):
    scene(tmp_path / "a.jpg")
    r = run("a.jpg", "--preset", "punchy", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "graded" / "a_graded.jpg"
    assert out.exists()
    before, after = np.asarray(Image.open(tmp_path / "a.jpg")), np.asarray(Image.open(out))
    assert before.shape == after.shape and not np.array_equal(before, after)


def test_never_replaces_the_original(tmp_path):
    original = scene(tmp_path / "a.jpg").read_bytes()
    r = run("a.jpg", "-o", ".", "--suffix", "", "--overwrite", cwd=tmp_path)
    assert r.returncode == 1 and "replace the original" in r.stderr
    assert (tmp_path / "a.jpg").read_bytes() == original


def test_existing_output_needs_overwrite(tmp_path):
    scene(tmp_path / "a.jpg")
    assert run("a.jpg", cwd=tmp_path).returncode == 0
    r = run("a.jpg", cwd=tmp_path)
    assert r.returncode == 1 and "--overwrite" in r.stderr
    assert run("a.jpg", "--overwrite", cwd=tmp_path).returncode == 0


def test_output_carries_no_gps_or_camera_metadata(tmp_path):
    make_jpeg(tmp_path / "geo.jpg", gps=True)
    assert Image.open(tmp_path / "geo.jpg").getexif().get(0x8825)
    assert run("geo.jpg", cwd=tmp_path).returncode == 0
    out = Image.open(tmp_path / "graded" / "geo_graded.jpg")
    assert not out.getexif() and out.info.get("icc_profile")


def test_folder_input_skips_broken_files_and_keeps_going(tmp_path):
    (tmp_path / "in").mkdir()
    scene(tmp_path / "in" / "ok.jpg")
    (tmp_path / "in" / "broken.jpg").write_bytes(b"not a jpeg")
    (tmp_path / "in" / "notes.txt").write_text("ignored")
    r = run("in", cwd=tmp_path)
    assert r.returncode == 1 and "broken.jpg: skipped" in r.stderr and "1 of 2" in r.stderr
    assert (tmp_path / "graded" / "ok_graded.jpg").exists()


def test_recursive_finds_photos_in_subfolders(tmp_path):
    (tmp_path / "in" / "deep").mkdir(parents=True)
    scene(tmp_path / "in" / "deep" / "x.png")
    assert "no photos found" in run("in", cwd=tmp_path).stderr
    assert run("in", "--recursive", cwd=tmp_path).returncode == 0
    assert (tmp_path / "graded" / "x_graded.jpg").exists()


def test_heic_input_and_tiff16_output(tmp_path):
    make_heic(tmp_path / "phone.heic")
    assert run("phone.heic", "--format", "tiff16", cwd=tmp_path).returncode == 0
    import cv2
    tif = cv2.imread(str(tmp_path / "graded" / "phone_graded.tif"), cv2.IMREAD_UNCHANGED)
    assert tif.dtype == np.uint16 and tif.shape == (200, 300, 3)


def test_before_after_sheet(tmp_path):
    scene(tmp_path / "a.jpg")
    assert run("a.jpg", "--before-after", cwd=tmp_path).returncode == 0
    sheet = Image.open(tmp_path / "graded" / "a_graded_before_after.jpg")
    assert sheet.width > 2 * 240 and sheet.height > 160


@pytest.mark.parametrize("settings,message", [
    ({"look": {"contrast": 9}}, "contrast"),
    ({"preset": "loud"}, "preset 'loud'"),
    ({"colour": {}}, "unknown section 'colour'"),
    ({"regions": {"space": {}}}, "unknown region 'space'"),
    ([1, 2], "must be a JSON object"),
    ({"label": "cat"}, "label 'cat'"),
])
def test_bad_settings_are_rejected_with_a_reason(tmp_path, settings, message):
    scene(tmp_path / "a.jpg")
    (tmp_path / "s.json").write_text(json.dumps(settings))
    r = run("a.jpg", "--settings", "s.json", cwd=tmp_path)
    assert r.returncode == 2 and message in r.stderr
    assert not (tmp_path / "graded").exists()


def test_settings_override_the_preset_key_by_key(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"preset": "bold", "look": {"vibrance": 0.1, "curve": {"strength": 0.2}},
                             "finish": {"dehaze": 0.05}}))
    s = load_settings(p, "natural")
    assert s.preset == "bold" and s.look["vibrance"] == 0.1
    assert s.look["curve"]["strength"] == 0.2 and s.look["curve"]["black"] == 0.035  # rest of bold's curve kept
    assert s.own_finish == {"dehaze": 0.05}


def test_explicit_dehaze_is_used_as_written(tmp_path, monkeypatch):
    import photo_grade.develop as develop
    import photo_grade.grade as grade
    seen = {}
    real_apply = grade.apply

    def spy(rgb, look, finish, *a, **k):
        seen.setdefault("dehaze", []).append(finish["dehaze"])
        return real_apply(rgb, look, finish, *a, **k)
    monkeypatch.setattr(grade, "apply", spy)
    monkeypatch.setattr(develop, "haze_factor", lambda rgb: 0.0)
    scene(tmp_path / "a.jpg")
    (tmp_path / "s.json").write_text(json.dumps({"finish": {"dehaze": 0.2}}))
    monkeypatch.chdir(tmp_path)
    assert main(["a.jpg", "--preset", "punchy", "--settings", "s.json", "-q"]) == 0
    assert seen["dehaze"] == [0.2]
    seen.clear()
    assert main(["a.jpg", "--preset", "punchy", "-q", "--overwrite"]) == 0
    assert seen["dehaze"] == [0.0], "a preset's dehaze is scaled by the measured haze"


def test_presets_command_prints_valid_json(tmp_path):
    r = run("presets", cwd=tmp_path)
    assert r.returncode == 0 and set(json.loads(r.stdout)) == {"natural", "punchy", "bold"}


def test_person_settings_are_reported_as_ignored(tmp_path):
    scene(tmp_path / "a.jpg")
    (tmp_path / "s.json").write_text(json.dumps({"finish": {"person": {"eyes": 0.5}}}))
    r = run("a.jpg", "--settings", "s.json", cwd=tmp_path)
    assert r.returncode == 0 and "ignored" in r.stderr


def test_doctor_hides_home_paths_and_names(monkeypatch, tmp_path):
    fake_home = tmp_path / "Zoë Ångström"
    monkeypatch.setattr(doctor.Path, "home", lambda: fake_home)
    text = json.dumps(doctor.collect(), ensure_ascii=False) + doctor.redact(
        rf"{fake_home}/x C:\Users\Roberto\AppData /home/someone/lib owner@example.com API_KEY=abc ZOË ÅNGSTRÖM")
    for leak in ("Zoë", "Roberto", "someone", "owner@example.com", "abc", "ZOË"):
        assert leak not in text, leak
    assert json.loads(run("doctor", "--json", cwd=tmp_path).stdout)["heic_support"] == "ok"


def test_food_label_lets_the_food_region_act(tmp_path):
    arr = np.full((160, 240, 3), (60, 60, 60), np.uint8)
    arr[50:110, 80:160] = (190, 120, 60)  # a salient "dish" in the middle
    Image.fromarray(arr).save(tmp_path / "dish.png")
    region = {"regions": {"food": {"vibrance": 0.6, "contrast": 0.3}}, "preset": "none"}
    (tmp_path / "plain.json").write_text(json.dumps(region))
    (tmp_path / "food.json").write_text(json.dumps({**region, "label": "food"}))
    for name in ("plain", "food"):
        assert run("dish.png", "--settings", f"{name}.json", "--suffix", f"_{name}", cwd=tmp_path).returncode == 0
    plain = np.asarray(Image.open(tmp_path / "graded" / "dish_plain.jpg"), np.float32)
    food = np.asarray(Image.open(tmp_path / "graded" / "dish_food.jpg"), np.float32)
    assert np.abs(food[60:100, 90:150] - plain[60:100, 90:150]).mean() > 3, "the dish should change only with the label"


# ---- review findings: each test reproduces a reported bug ----

def _case_insensitive(d):
    probe = d / "CaseProbe"
    probe.write_text("x")
    try:
        return (d / "caseprobe").exists()
    finally:
        probe.unlink()


def test_original_safe_when_only_the_letter_case_differs(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("this filesystem is case-sensitive, so the names are different files")
    original = scene(tmp_path / "C.JPG").read_bytes()
    r = run("C.JPG", "-o", ".", "--suffix", "", "--overwrite", cwd=tmp_path)
    assert r.returncode == 1 and "replace the original" in r.stderr
    assert (tmp_path / "C.JPG").read_bytes() == original


def test_original_safe_when_the_output_is_a_hard_link_to_it(tmp_path):
    import os
    original = scene(tmp_path / "h.jpg").read_bytes()
    (tmp_path / "o").mkdir()
    os.link(tmp_path / "h.jpg", tmp_path / "o" / "h_graded.jpg")
    r = run("h.jpg", "-o", "o", "--overwrite", cwd=tmp_path)
    assert r.returncode == 1 and "replace the original" in r.stderr
    assert (tmp_path / "h.jpg").read_bytes() == original


def test_inputs_that_would_share_an_output_name_stop_before_grading(tmp_path):
    scene(tmp_path / "a.jpg")
    scene(tmp_path / "a.png")
    r = run("a.jpg", "a.png", "--overwrite", cwd=tmp_path)
    assert r.returncode == 2 and "same output" in r.stderr
    assert not (tmp_path / "graded").exists()


def test_the_same_file_twice_is_graded_once(tmp_path):
    scene(tmp_path / "a.jpg")
    r = run("a.jpg", "a.jpg", cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.count("a_graded.jpg") == 1


def test_a_decompression_bomb_is_skipped_and_the_batch_continues(tmp_path, monkeypatch):
    from PIL import Image as PILImage
    scene(tmp_path / "big.png", size=(240, 160))
    scene(tmp_path / "ok.jpg", size=(20, 10))
    import photo_grade.imageio  # noqa: F401  (sets the limit first, so ours wins)
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 5000)
    monkeypatch.chdir(tmp_path)
    assert main(["big.png", "ok.jpg", "-q"]) == 1
    assert (tmp_path / "graded" / "ok_graded.jpg").exists()


def test_16_bit_grey_tiff_keeps_its_brightness(tmp_path):
    arr = np.full((60, 80), int(0.46 * 65535), np.uint16)
    im = Image.fromarray(arr)
    assert im.mode == "I;16"
    im.save(tmp_path / "grey16.tif")
    assert run("grey16.tif", "--preset", "none", cwd=tmp_path).returncode == 0
    out = np.asarray(Image.open(tmp_path / "graded" / "grey16_graded.jpg"), np.float32)
    assert 90 < out.mean() < 170, out.mean()


def test_before_after_sheet_is_protected_like_other_outputs(tmp_path):
    scene(tmp_path / "a.jpg")
    assert run("a.jpg", "--before-after", cwd=tmp_path).returncode == 0
    (tmp_path / "graded" / "a_graded.jpg").unlink()
    r = run("a.jpg", "--before-after", cwd=tmp_path)
    assert r.returncode == 1 and "a_graded_before_after.jpg already exists" in r.stderr
    sheet = Image.open(tmp_path / "graded" / "a_graded_before_after.jpg")
    assert sheet.info.get("icc_profile")


def test_outputs_inside_the_input_folder_are_not_graded_again(tmp_path):
    (tmp_path / "trip").mkdir()
    scene(tmp_path / "trip" / "a.jpg")
    for _ in range(2):
        assert run("trip", "-o", "trip/graded", "--recursive", "--overwrite", cwd=tmp_path).returncode == 0
    assert sorted(p.name for p in (tmp_path / "trip" / "graded").iterdir()) == ["a_graded.jpg"]


def test_no_half_written_files_are_left_behind(tmp_path):
    scene(tmp_path / "a.jpg")
    assert run("a.jpg", "--before-after", cwd=tmp_path).returncode == 0
    assert not [p for p in (tmp_path / "graded").iterdir() if ".part" in p.name]


@pytest.mark.parametrize("args", [["doctor", "--bogus"], ["presets", "extra"]])
def test_subcommands_reject_unexpected_arguments(tmp_path, args):
    r = run(*args, cwd=tmp_path)
    assert r.returncode == 2 and "unexpected" in r.stderr
