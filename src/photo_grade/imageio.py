"""Reading photos as sRGB float arrays (JPEG, PNG, TIFF, HEIC) and writing JPEGs without camera metadata."""
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from pi_heif import register_heif_opener

register_heif_opener()
# iPhone panoramas reach ~63 MP and a 120 MP file must open quietly. Pillow warns above this and refuses above
# twice it (260 MP), which keeps its decompression-bomb protection.
Image.MAX_IMAGE_PIXELS = 130_000_000
_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
_SRGB = _SRGB_PROFILE.tobytes()

# What a photo that can't be read, decoded or held in memory raises (a half-downloaded cloud-synced file included).
# Pillow's DecompressionBombError (an image claiming far more pixels than the limit) is a read error too.
READ_ERRORS = (OSError, cv2.error, MemoryError, Image.DecompressionBombError)


def read_error_why(e: BaseException) -> str:
    if isinstance(e, Image.DecompressionBombError):
        return "it claims more pixels than photo-grade will open (a possible decompression bomb)"
    if isinstance(e, MemoryError):
        return "it's too large to process with the memory available"
    if isinstance(e, cv2.error):
        return "its image data couldn't be processed"
    return e.strerror or str(e) or type(e).__name__


def to_srgb(im: Image.Image, path) -> Image.Image:
    """Camera originals (iPhone HEIC) are usually Display P3; read as-is they'd look duller once tagged sRGB."""
    icc = im.info.get("icc_profile")
    if im.mode.startswith("I"):
        # 16-bit (and 32-bit integer) greyscale: scale to 8 bits; a plain convert("RGB") would clip it to white.
        top = 65535 if im.mode.startswith("I;16") or np.asarray(im).max() <= 65535 else np.asarray(im).max()
        im = Image.fromarray(np.clip(np.asarray(im, np.float64) * (255 / top), 0, 255).astype(np.uint8), "L")
    im = im.convert("RGB")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if src.profile.xcolor_space.strip() != "RGB" or "sRGB" in ImageCms.getProfileDescription(src):
            return im
        return ImageCms.profileToProfile(im, src, _SRGB_PROFILE, ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                         outputMode="RGB")
    except (ImageCms.PyCMSError, OSError) as e:
        print(f"Warning: {Path(path).name} has an unreadable colour profile ({e}); treating it as sRGB.",
              file=sys.stderr)
        return im


def oriented_size(path) -> tuple[int, int]:
    """(width, height) as the photo displays, from the header and EXIF orientation only (no decode)."""
    with Image.open(path) as im:
        w, h = im.size
        return (h, w) if im.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)


def load_rgb(path, max_side: int | None = None) -> np.ndarray:
    with Image.open(path) as im:
        im = to_srgb(ImageOps.exif_transpose(im), path)
        if max_side and max(im.size) > max_side:
            k = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        u8 = np.asarray(im)
    del im
    # Via uint8 and in place: a 24 MP photo would otherwise hold ~1 GB of float temporaries at once.
    arr = u8.astype(np.float32)
    del u8
    arr /= 255.0
    return arr


def _umask() -> int:
    """The process umask (reading it requires setting it, so it is restored at once)."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def save_image_jpeg(im: Image.Image, path, quality: int = 95) -> Path:
    """Write a JPEG with an sRGB profile and no EXIF, atomically: to a temporary file in the same folder, then renamed
    over the destination. A crash never leaves a half-written file, and renaming replaces the directory entry rather
    than writing into whatever file it pointed at."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}.", suffix=".part.jpg")
    os.close(fd)
    try:
        os.chmod(tmp, 0o666 & ~_umask())  # mkstemp makes the file private (0600); outputs get normal permissions
        im.save(tmp, "JPEG", quality=quality, icc_profile=_SRGB)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def save_jpeg(rgb: np.ndarray, path, quality: int = 95) -> Path:
    arr = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    return save_image_jpeg(Image.fromarray(arr, "RGB"), path, quality)
