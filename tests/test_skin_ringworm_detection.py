import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _detect_common_skin_disease, _detect_tinea_corporis


def test_detects_ring_like_lesion():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cx, cy = 32, 32
    for y in range(64):
        for x in range(64):
            r = np.hypot(x - cx, y - cy)
            if 10 <= r <= 20:
                img[y, x] = [220, 70, 70]
            elif r < 8:
                img[y, x] = [220, 220, 200]
            else:
                img[y, x] = [245, 238, 236]

    detected, score = _detect_tinea_corporis(img)
    assert bool(detected) is True
    assert score > 0.2
    result = _detect_common_skin_disease(img)
    assert result is not None and 'Ringworm' in result['disease']


def test_detects_eczema_like_patch():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if 10 <= x <= 52 and 10 <= y <= 52:
                img[y, x] = [180, 100, 100]
            else:
                img[y, x] = [240, 230, 230]

    result = _detect_common_skin_disease(img)
    assert result is not None and 'Eczema' in result['disease']


def test_rejects_non_ring_lesion():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = [180, 180, 170]
    img[18:46, 18:46] = [200, 80, 80]

    detected, score = _detect_tinea_corporis(img)
    assert bool(detected) is False
    assert score < 0.3
