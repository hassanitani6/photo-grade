# Changelog

Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed
- Graded JPEGs and before/after previews are now created with the usual permissions for new files instead of
being readable only by your user account.

## 0.1.0 — unreleased

First public release: natural, punchy and bold presets; haze-aware dehaze; look, tone curve, HSL, colour wheels,
region looks and finish ops; JPEG/PNG/TIFF/HEIC input; JPEG and 16-bit TIFF output; before/after previews;
`presets` and `doctor` commands. Originals are never overwritten (including
case-only name differences and hard links); outputs are written atomically.
