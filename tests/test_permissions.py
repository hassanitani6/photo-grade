import os
import sys

import pytest
from PIL import Image

from photo_grade.imageio import save_image_jpeg


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file permissions")
@pytest.mark.parametrize("umask, expected", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_jpeg_outputs_get_the_usual_permissions(tmp_path, umask, expected):
    dest = tmp_path / "photo_graded.jpg"
    old = os.umask(umask)
    try:
        save_image_jpeg(Image.new("RGB", (16, 12), (120, 90, 60)), dest)
    finally:
        os.umask(old)
    assert dest.stat().st_mode & 0o777 == expected
