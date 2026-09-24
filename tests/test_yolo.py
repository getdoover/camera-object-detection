"""Tests for the hand-rolled YOLO pre/post-processing.

This code replaces ultralytics, so the geometry has to be right -- a silent
half-box offset here would show up only as slightly-wrong annotations.
"""

import numpy as np
import pytest
from common.yolo import Detection, YoloOnnx, letterbox


class TestLetterbox:
    def test_square_output(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, scale, (left, top) = letterbox(img, 640)
        assert out.shape == (640, 640, 3)
        assert scale == 1.0
        # 640x480 scaled to fit 640 -> 640x480, so 80px of padding top and bottom.
        assert (left, top) == (0, 80)

    def test_preserves_aspect_ratio(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out, scale, (left, top) = letterbox(img, 640)
        assert out.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 1920)
        assert left == 0
        assert top == (640 - round(1080 * scale)) // 2

    def test_portrait(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        out, _, (left, top) = letterbox(img, 640)
        assert out.shape == (640, 640, 3)
        assert top == 0
        assert left == 160

    def test_roundtrip_maps_a_box_back(self):
        """A box placed in letterboxed space must map back to where it started."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, scale, (left, top) = letterbox(img, 640)

        original = (400, 300, 900, 800)
        x1, y1, x2, y2 = original
        # Forward into letterbox space...
        padded = (
            x1 * scale + left,
            y1 * scale + top,
            x2 * scale + left,
            y2 * scale + top,
        )
        # ...and back out, the way detect() does.
        back = tuple(
            round((v - pad) / scale) for v, pad in zip(padded, (left, top, left, top))
        )
        assert back == original


class TestDecode:
    # _decode tells the anchor axis from the class axis by length, which is only
    # meaningful when anchors outnumber 4+num_classes -- true of every real export
    # (8400 anchors vs at most 84 rows). Pad the fakes out so they exercise the same
    # branch a real model would.
    ANCHORS = 64

    @classmethod
    def _output(cls, boxes, scores, num_classes=3):
        """Build a fake (1, 4+nc, anchors) model output."""
        pred = np.zeros((1, 4 + num_classes, cls.ANCHORS), dtype=np.float32)
        for i, ((cx, cy, w, h), (cid, score)) in enumerate(zip(boxes, scores)):
            pred[0, :4, i] = (cx, cy, w, h)
            pred[0, 4 + cid, i] = score
        return pred

    def test_converts_centre_to_top_left(self):
        """cv2.dnn.NMSBoxes reads its input as top-left, so decode must convert.

        Passing centre-form boxes shifts each by half its own size, which perturbs
        IoU differently for large and small boxes and mis-suppresses.
        """
        out = self._output([(100, 100, 40, 20)], [(0, 0.9)])
        boxes, scores, class_ids = YoloOnnx._decode(out, 0.5)
        assert boxes == [[80.0, 90.0, 40.0, 20.0]]
        assert scores == [pytest.approx(0.9)]
        assert list(class_ids) == [0]

    def test_filters_below_confidence(self):
        out = self._output([(10, 10, 4, 4), (20, 20, 4, 4)], [(0, 0.9), (1, 0.1)])
        boxes, scores, _ = YoloOnnx._decode(out, 0.5)
        assert len(boxes) == 1
        assert scores == [pytest.approx(0.9)]

    def test_empty_when_nothing_passes(self):
        out = self._output([(10, 10, 4, 4)], [(0, 0.05)])
        assert YoloOnnx._decode(out, 0.5) == ([], [], [])

    def test_picks_the_highest_scoring_class(self):
        out = self._output([(10, 10, 4, 4)], [(0, 0.4)])
        out[0, 4 + 2, 0] = 0.8
        _, scores, class_ids = YoloOnnx._decode(out, 0.5)
        assert list(class_ids) == [2]
        assert scores == [pytest.approx(0.8)]

    def test_handles_transposed_output(self):
        """Some exporters emit (1, anchors, 4+nc) instead."""
        straight = self._output([(100, 100, 40, 20)], [(0, 0.9)])

        a = YoloOnnx._decode(straight, 0.5)
        b = YoloOnnx._decode(straight.transpose(0, 2, 1), 0.5)
        assert a[0] == b[0]
        assert a[1] == b[1]


class TestDetection:
    def test_area(self):
        assert Detection("x", 1.0, (0, 0, 10, 20)).area == 200

    def test_area_of_an_inverted_box_is_zero(self):
        assert Detection("x", 1.0, (10, 10, 5, 5)).area == 0

    def test_to_dict(self):
        d = Detection("person", 0.87654, (1, 2, 3, 4)).to_dict()
        assert d == {"label": "person", "confidence": 0.877, "box": [1, 2, 3, 4]}
