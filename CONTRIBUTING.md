# Contributing

Thanks for helping! Small, focused pull requests are easiest to review.

## Set up

```bash
uv sync
uv run pytest
```

## Ground rules

- Bug fixes come with a test that fails before the fix and passes after it.
- Test images are generated in code (see `tests/gradekit.py`) — please don't add real photos to the repository.
- Keep every finish and develop operation (curve, HSL, wheels, light, dehaze…) a no-op at its default; the base
  look defaults (contrast 0.10, vibrance 0.20, warmth 0.025) are a deliberate gentle grade. Add new settings to the
  range tables in `looks.py` so they are validated.
- No network access or telemetry, and no new required dependencies without an issue first.

## Where things are

`grade.py` runs the whole grade in a fixed order (`apply`). `develop.py` holds dehaze, curve, HSL, colour wheels
and local light; `regions.py` the region masks and region looks; `looks.py` the settings, ranges and presets;
`imageio.py` reading and writing; `cli.py` the command line.

The engine can also enhance faces and protect skin around them (`regions.Subject` with face boxes), but the command
line does not detect faces, so those paths are reached only from the tests and from Python code that supplies boxes.
