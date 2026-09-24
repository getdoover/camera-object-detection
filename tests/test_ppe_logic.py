"""Tests for the PPE compliance reasoning.

The model's raw output is faked here on purpose: what's worth testing is the
attribution of equipment to people and the compliance decision, not whether a
particular set of weights fires on a particular JPEG.
"""

from types import SimpleNamespace

import pytest
from common.detectors.ppe import (
    CONTAINMENT_THRESHOLD,
    Person,
    PPEDetector,
    _containment,
)
from common.yolo import Detection


def cfg(hard_hat=True, high_vis=True, confidence=40):
    v = lambda value: SimpleNamespace(value=value)
    return SimpleNamespace(
        require_hard_hat=v(hard_hat),
        require_high_vis=v(high_vis),
        confidence=v(confidence),
    )


def det(label, box, conf=0.9):
    return Detection(label, conf, box)


PERSON_BOX = (100, 100, 200, 400)


class TestContainment:
    def test_fully_inside(self):
        hat = det("hardhat", (120, 100, 180, 140))
        person = det("person", PERSON_BOX)
        assert _containment(hat, person) == 1.0

    def test_disjoint(self):
        hat = det("hardhat", (500, 500, 560, 540))
        person = det("person", PERSON_BOX)
        assert _containment(hat, person) == 0.0

    def test_half_inside(self):
        # Box spans the person's left edge, so half its area is outside.
        hat = det("hardhat", (50, 100, 150, 140))
        person = det("person", PERSON_BOX)
        assert _containment(hat, person) == pytest.approx(0.5)

    def test_iou_would_fail_here(self):
        """A hard hat is tiny next to a person, so IoU is near zero even when worn.

        This is the reason containment is used instead of IoU -- guard against
        somebody "simplifying" it back.
        """
        hat = det("hardhat", (120, 100, 180, 140))
        person = det("person", PERSON_BOX)
        hat_area, person_area = hat.area, person.area
        iou = hat_area / (hat_area + person_area - hat_area)
        assert iou < 0.1
        assert _containment(hat, person) == 1.0


class TestAssignment:
    def test_positive_hat_marks_compliant(self):
        person = Person(det("person", PERSON_BOX))
        assert PPEDetector._assign(det("hardhat", (120, 100, 180, 140)), [person])
        assert person.hard_hat is True

    def test_negative_hat_marks_violation(self):
        person = Person(det("person", PERSON_BOX))
        assert PPEDetector._assign(det("no-hardhat", (120, 100, 180, 140)), [person])
        assert person.hard_hat is False

    def test_positive_wins_over_negative(self):
        """These models routinely emit both classes over one head.

        Treating that as a violation is a false alarm, so a positive detection must
        win regardless of the order they arrive in.
        """
        person = Person(det("person", PERSON_BOX))
        PPEDetector._assign(det("hardhat", (120, 100, 180, 140)), [person])
        PPEDetector._assign(det("no-hardhat", (120, 100, 180, 140)), [person])
        assert person.hard_hat is True

    def test_negative_does_not_overwrite_positive_in_either_order(self):
        person = Person(det("person", PERSON_BOX))
        PPEDetector._assign(det("no-hardhat", (120, 100, 180, 140)), [person])
        PPEDetector._assign(det("hardhat", (120, 100, 180, 140)), [person])
        assert person.hard_hat is True

    def test_unassigned_when_nobody_encloses_it(self):
        person = Person(det("person", PERSON_BOX))
        assert not PPEDetector._assign(det("hardhat", (900, 900, 950, 950)), [person])
        assert person.hard_hat is None

    def test_goes_to_the_most_enclosing_person(self):
        near = Person(det("person", (100, 100, 200, 400)))
        far = Person(det("person", (150, 100, 400, 400)))
        # Sits wholly inside `near`, only partly inside `far`.
        PPEDetector._assign(det("hardhat", (105, 100, 145, 140)), [near, far])
        assert near.hard_hat is True
        assert far.hard_hat is None

    def test_threshold_is_respected(self):
        person = Person(det("person", PERSON_BOX))
        # 25% inside -- below the threshold, so unattributed.
        item = det("hardhat", (25, 100, 125, 140))
        assert _containment(item, person.detection) < CONTAINMENT_THRESHOLD
        assert not PPEDetector._assign(item, [person])

    def test_vest_classes(self):
        person = Person(det("person", PERSON_BOX))
        PPEDetector._assign(det("safety vest", (110, 180, 190, 300)), [person])
        assert person.high_vis is True

        other = Person(det("person", PERSON_BOX))
        PPEDetector._assign(det("no-safety vest", (110, 180, 190, 300)), [other])
        assert other.high_vis is False


class TestViolations:
    def test_unknown_counts_as_missing_when_required(self):
        """If we're asked to police hard hats and can't see one, that's the finding."""
        person = Person(det("person", PERSON_BOX))
        assert person.hard_hat is None
        assert person.violations(True, False) == ["hard_hat"]

    def test_unknown_is_ignored_when_not_required(self):
        person = Person(det("person", PERSON_BOX))
        assert person.violations(False, False) == []

    def test_compliant_person_has_no_violations(self):
        person = Person(det("person", PERSON_BOX))
        person.hard_hat = True
        person.high_vis = True
        assert person.violations(True, True) == []

    def test_both_missing(self):
        person = Person(det("person", PERSON_BOX))
        person.hard_hat = False
        person.high_vis = False
        assert person.violations(True, True) == ["hard_hat", "high_vis"]

    def test_only_the_required_check_is_reported(self):
        person = Person(det("person", PERSON_BOX))
        person.hard_hat = False
        person.high_vis = False
        assert person.violations(True, False) == ["hard_hat"]
        assert person.violations(False, True) == ["high_vis"]


class TestImpliedPerson:
    def test_grows_downward_from_a_bare_head(self):
        item = det("no-hardhat", (100, 100, 140, 140))
        person = PPEDetector._imply_person(item, (1080, 1920))
        x1, y1, x2, y2 = person.detection.box
        assert person.implied is True
        # Top edge is unchanged -- a head is at the top of a body.
        assert y1 == 100
        assert y2 > 140
        assert x1 < 100 and x2 > 140

    def test_clamped_to_the_frame(self):
        item = det("no-hardhat", (0, 0, 40, 40))
        x1, y1, x2, y2 = PPEDetector._imply_person(item, (50, 50)).detection.box
        assert (x1, y1) == (0, 0)
        assert x2 <= 50 and y2 <= 50
