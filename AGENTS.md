# Notes for coding agents

- Install: `uv sync`. Test: `uv run pytest` (a few seconds).
- Package code is in `src/photo_grade/`; `grade.apply` is the single entry point for a grade.
- Every change needs a test; test images are synthetic (`tests/gradekit.py`, `tests/conftest.py`).
- Finish/develop ops must be no-ops at their defaults; every setting must have a range in `looks.py`.
- Do not add network calls, telemetry, or copying of EXIF/GPS metadata into outputs.
