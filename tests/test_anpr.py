"""Tests for the plate OCR result handling.

The OCR library's return shape has changed across versions and its input format is
strict, and both broke this app during development:

* passing a grayscale crop makes the ONNX session raise "invalid dimensions for
  input" -- every plate read comes back empty rather than degrading
* 1.x returns ``[PlatePrediction(...)]``, not the list of strings older versions
  returned, so reading ``texts[0]`` as a string yields the dataclass repr

Both are pinned here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from common.detectors.anpr import (
    MIN_CROP_WIDTH,
    ANPRDetector,
    ANPRResult,
    Plate,
)
from common.yolo import Detection


def cfg(min_chars=4, confidence=40):
    v = lambda value: SimpleNamespace(value=value)
    return SimpleNamespace(min_plate_chars=v(min_chars), confidence=v(confidence))


def detector(ocr):
    """An ANPRDetector with the models bypassed -- only OCR handling under test."""
    d = ANPRDetector.__new__(ANPRDetector)
    d.config = cfg()
    d.ocr = ocr
    return d


class FakeOCR:
    def __init__(self, result):
        self.result = result
        self.seen = None

    def run(self, crop):
        self.seen = crop
        return self.result


class TestRunOcrReturnShapes:
    def test_plate_prediction_objects(self):
        """fast-plate-ocr 1.x -- what the shipped version actually returns."""
        pred = SimpleNamespace(plate="AD799KB", char_probs=None)
        text, conf = detector(FakeOCR([pred]))._run_ocr(np.zeros((10, 10, 3)))
        assert text == "AD799KB"
        assert conf is None

    def test_plate_prediction_with_char_probs(self):
        pred = SimpleNamespace(plate="KRW301", char_probs=[0.99, 0.8, 0.95])
        text, conf = detector(FakeOCR([pred]))._run_ocr(np.zeros((10, 10, 3)))
        assert text == "KRW301"
        # The weakest character, not the mean -- one bad digit is what makes a
        # plate wrong.
        assert conf == pytest.approx(0.8)

    def test_bare_list_of_strings(self):
        text, _ = detector(FakeOCR(["ABC123"]))._run_ocr(np.zeros((10, 10, 3)))
        assert text == "ABC123"

    def test_texts_and_confidences_tuple(self):
        ocr = FakeOCR((["XYZ789"], [[0.9, 0.7]]))
        text, conf = detector(ocr)._run_ocr(np.zeros((10, 10, 3)))
        assert text == "XYZ789"
        assert conf == pytest.approx(0.7)

    def test_bare_string(self):
        text, _ = detector(FakeOCR("ABC123"))._run_ocr(np.zeros((10, 10, 3)))
        assert text == "ABC123"

    def test_empty_result(self):
        text, _ = detector(FakeOCR([]))._run_ocr(np.zeros((10, 10, 3)))
        assert text is None

    def test_nested_char_probs_are_flattened(self):
        pred = SimpleNamespace(plate="AA11", char_probs=[[0.9, 0.5], [0.99]])
        _, conf = detector(FakeOCR([pred]))._run_ocr(np.zeros((10, 10, 3)))
        assert conf == pytest.approx(0.5)

    def test_unusable_char_probs_are_ignored(self):
        pred = SimpleNamespace(plate="AA11", char_probs="not-numbers")
        _, conf = detector(FakeOCR([pred]))._run_ocr(np.zeros((10, 10, 3)))
        assert conf is None


class TestRead:
    """_read layers cleanup and the length guard on top of _run_ocr."""

    @staticmethod
    def _read(result, min_chars=4):
        d = detector(FakeOCR(result))
        d.config = cfg(min_chars=min_chars)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        return d._read(image, Detection("license_plate", 0.9, (10, 10, 150, 60)))

    def test_strips_separators_and_uppercases(self):
        pred = SimpleNamespace(plate="ad-799 kb", char_probs=None)
        assert self._read([pred])[0] == "AD799KB"

    def test_rejects_too_short(self):
        """A short read is nearly always a partial plate, not a real one."""
        pred = SimpleNamespace(plate="AB", char_probs=None)
        assert self._read([pred])[0] is None

    def test_accepts_at_the_boundary(self):
        pred = SimpleNamespace(plate="ABCD", char_probs=None)
        assert self._read([pred], min_chars=4)[0] == "ABCD"

    def test_ocr_failure_does_not_propagate(self):
        """A bad frame must not take down the whole analysis."""

        class Boom:
            def run(self, crop):
                raise RuntimeError("onnx said no")

        d = detector(Boom())
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        assert d._read(image, Detection("license_plate", 0.9, (10, 10, 150, 60))) == (
            None,
            None,
        )

    def test_no_ocr_model_returns_nothing(self):
        d = detector(None)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        assert d._read(image, Detection("license_plate", 0.9, (10, 10, 150, 60))) == (
            None,
            None,
        )


class TestCrop:
    def test_keeps_three_channels(self):
        """Grayscale fails the OCR session outright -- see the module docstring."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = ANPRDetector._crop(image, Detection("p", 0.9, (10, 10, 150, 60)))
        assert crop is not None
        assert crop.ndim == 3
        assert crop.shape[2] == 3

    def test_pads_the_box(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        box = (50, 50, 150, 90)
        crop = ANPRDetector._crop(image, Detection("p", 0.9, box))
        # Padded, so wider than the raw box.
        assert crop.shape[1] > box[2] - box[0]

    def test_clamps_to_the_frame(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = ANPRDetector._crop(image, Detection("p", 0.9, (0, 0, 100, 40)))
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100

    def test_rejects_a_crop_too_small_to_read(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        narrow = (10, 10, 10 + MIN_CROP_WIDTH - 20, 30)
        assert ANPRDetector._crop(image, Detection("p", 0.9, narrow)) is None


class TestResults:
    def test_unread_plate_is_still_reported(self):
        """A detected-but-unreadable plate is real information, and the annotated
        image draws it -- dropping it would make image and payload disagree."""
        result = ANPRResult([Plate(Detection("p", 0.9, (1, 2, 3, 4)), None)])
        assert result.has_findings
        assert result.read_plates == []
        assert result.to_dict()["plates"][0]["plate"] is None

    def test_label_falls_back(self):
        assert Plate(Detection("p", 0.9, (1, 2, 3, 4))).label == "plate"
        assert Plate(Detection("p", 0.9, (1, 2, 3, 4)), "ABC123").label == "ABC123"

    def test_ocr_confidence_omitted_when_absent(self):
        d = Plate(Detection("p", 0.9, (1, 2, 3, 4)), "ABC123").to_dict()
        assert "ocr_confidence" not in d

    def test_ocr_confidence_included_when_present(self):
        d = Plate(Detection("p", 0.9, (1, 2, 3, 4)), "ABC123", 0.87654).to_dict()
        assert d["ocr_confidence"] == 0.877
