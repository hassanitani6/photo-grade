import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


def make_jpeg(path, size=(400, 200), orientation=None, gps=False, color=(200, 90, 40)):
    im = Image.new("RGB", size, color)
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    if gps:
        # Synthetic coordinates in the open Atlantic, so a test photo never points anywhere real.
        exif[0x8825] = {1: "N", 2: (10.0, 0.0, 0.0), 3: "W", 4: (30.0, 0.0, 0.0)}
    im.save(path, "JPEG", exif=exif)
    return path


def make_heic(path, size=(300, 200)):
    """A HEIC test file. Writing HEIC needs the full pillow-heif (a test-only dependency); the tool itself only
    reads HEIC, through the decode-only pi-heif."""
    from pillow_heif import from_pillow
    from_pillow(Image.new("RGB", size, (200, 90, 40))).save(path, quality=90)
    return path
