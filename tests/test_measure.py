import json

from pathlib import Path

import numpy as np

import pytest

from photo_grade import measure



from gradekit import NEUTRAL, hsv_image




def test_face_stats_report_the_red_channel_of_the_face_core():
    from photo_grade.regions import Subject, face_stats
    img = np.full((400, 600, 3), 0.4, np.float32)
    img[120:220, 260:340] = [0.99, 0.7, 0.6]
    got = face_stats(img, Subject((600, 400), people=((260, 120, 340, 220),)))
    assert abs(got[0]["red_p995"] - 0.99) < 1e-3


