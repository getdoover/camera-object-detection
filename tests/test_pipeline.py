"""Tests for the detector-agnostic pipeline: zones, summaries, events, notifications.

This is the logic both app shells used to carry separate copies of, so it is where a
behaviour change would show up on one variant and not the other. Real PPE / ANPR
detectors are used for the reporting half (their summaries and alerts are the contract
people read); their models are never loaded.
"""

from types import SimpleNamespace

import numpy as np

from common import pipeline
from common.detectors import load_enabled, wanted_for_reason
from common.detectors.anpr import ANPRDetector, ANPRResult, Plate
from common.detectors.ppe import Person, PPEDetector, PPEResult
from common.yolo import Detection

IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

LEFT_BOX = (10, 10, 50, 90)  # centre (30, 50) -> (0.15, 0.5)
RIGHT_BOX = (150, 10, 190, 90)  # centre (170, 50) -> (0.85, 0.5)


def v(value):
    return SimpleNamespace(value=value)


def zone(detectors, left=True, notify=True, name=None, id=1):
    x0, x1 = (0.0, 0.5) if left else (0.5, 1.0)
    return {
        "id": id,
        "kind": "intrusion",
        "detectors": detectors,
        "points": [[x0, 0.0], [x1, 0.0], [x1, 1.0], [x0, 1.0]],
        "notify": notify,
        "name": name,
    }


def violator(box, missing=("hard_hat",)):
    person = Person(Detection("person", 0.9, box))
    person.missing = list(missing)
    return person


def compliant(box):
    person = Person(Detection("person", 0.9, box))
    person.missing = []
    return person


class FakeDetector:
    """Wraps a real detector's reporting, returning a canned result from `analyse`."""

    def __init__(self, real, result):
        self._real = real
        self._result = result
        self.name = real.name
        self.camera_reasons = real.camera_reasons

    def analyse(self, image, size):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def __getattr__(self, attr):
        return getattr(self._real, attr)


def ppe(people=(), notify=True):
    real = PPEDetector.__new__(PPEDetector)
    real.config = SimpleNamespace(notify_on_violation=v(notify))
    result = PPEResult(list(people), [])
    result.violators = [p for p in people if p.missing]
    return FakeDetector(real, result)


def anpr(plates=(), notify=False):
    real = ANPRDetector.__new__(ANPRDetector)
    real.config = SimpleNamespace(notify_on_plate=v(notify))
    return FakeDetector(real, ANPRResult(list(plates)))


def plate(text, box=RIGHT_BOX):
    return Plate(Detection("plate", 0.8, box), text)


def run(detectors, zones=None, camera="Gate cam"):
    analysis = pipeline.analyse(detectors, IMAGE, 640, zones)
    return analysis, pipeline.report(detectors, [analysis], camera)


class TestSummary:
    """Ported from the shells' old `_summarise`, whose strings people already read."""

    def test_nothing(self):
        _, report = run([ppe(), anpr()])
        assert report.summary == "nothing detected"

    def test_one_violator(self):
        _, report = run([ppe([violator(LEFT_BOX)])])
        assert report.summary == "1 person missing hard hat"

    def test_several_violators_pluralise(self):
        _, report = run([ppe([violator(LEFT_BOX)] * 3)])
        assert report.summary == "3 people missing hard hat"

    def test_deduplicates_missing_items(self):
        people = [
            violator(LEFT_BOX, ["hard_hat", "high_vis"]),
            violator(LEFT_BOX, ["hard_hat"]),
        ]
        _, report = run([ppe(people)])
        assert report.summary == "2 people missing hard hat, high vis"

    def test_plates(self):
        _, report = run([anpr([plate("ABC123")])])
        assert report.summary == "plate(s) ABC123"

    def test_unread_plates_are_not_reported(self):
        _, report = run([anpr([plate(None)])])
        assert report.summary == "nothing detected"

    def test_both(self):
        _, report = run([ppe([violator(LEFT_BOX, ["high_vis"])]), anpr([plate("XYZ")])])
        assert report.summary == "1 person missing high vis; plate(s) XYZ"


class TestFindings:
    def test_findings_are_unfiltered(self):
        """Zones narrow what's reported, never what's published as seen."""
        zones = [zone(["ppe"], left=True)]
        analysis, report = run([ppe([violator(RIGHT_BOX)])], zones)
        assert len(analysis.findings["ppe"]["violations"]) == 1
        assert report.summary == "nothing detected"

    def test_compliant_person_counts_as_found(self):
        """Nothing to report, but still worth a timeline entry."""
        analysis, report = run([ppe([compliant(LEFT_BOX)])])
        assert analysis.found_anything
        assert report.summary == "nothing detected"

    def test_empty_frame_found_nothing(self):
        analysis, _ = run([ppe(), anpr()])
        assert not analysis.found_anything

    def test_failing_detector_does_not_take_others_down(self):
        broken = ppe()
        broken._result = RuntimeError("onnx exploded")
        analysis, report = run([broken, anpr([plate("ABC123")])])
        assert "ppe" not in analysis.findings
        assert report.summary == "plate(s) ABC123"

    def test_annotates_every_detector(self):
        analysis, _ = run([ppe([violator(LEFT_BOX)]), anpr([plate("ABC")])])
        drawn = analysis.annotate(IMAGE)
        assert drawn.shape == IMAGE.shape
        assert drawn.any()
        assert not IMAGE.any(), "annotate must draw on a copy"


class TestZones:
    def test_each_detector_filtered_by_its_own_zones(self):
        """A PPE zone says nothing about where plates matter."""
        zones = [zone(["ppe"], left=True)]
        _, report = run(
            [ppe([violator(LEFT_BOX), violator(RIGHT_BOX)]), anpr([plate("ABC")])],
            zones,
        )
        assert report.summary == "1 person missing hard hat; plate(s) ABC"

    def test_no_zones_means_whole_frame(self):
        _, report = run([ppe([violator(RIGHT_BOX)])], zones=[])
        assert report.summary == "1 person missing hard hat"

    def test_zones_for_other_detectors_dont_restrict(self):
        _, report = run([ppe([violator(RIGHT_BOX)])], [zone(["objects"], left=True)])
        assert report.summary == "1 person missing hard hat"


class TestNotifications:
    def test_config_switch_when_no_zones(self):
        _, report = run([ppe([violator(LEFT_BOX)], notify=True)])
        assert [n.text for n in report.notifications] == [
            "Gate cam detected someone without hard hat."
        ]
        assert report.notifications[0].severity == "warn"
        assert report.notifications[0].topic == "ppe_event"

    def test_switch_off_is_quiet(self):
        _, report = run([ppe([violator(LEFT_BOX)], notify=False)])
        assert report.notifications == []

    def test_zone_overrides_switch_and_names_itself(self):
        zones = [zone(["anpr"], left=False, notify=True, name="Entry")]
        _, report = run([anpr([plate("ABC")], notify=False)], zones)
        assert [n.text for n in report.notifications] == [
            "Gate cam read plate(s) ABC in Entry."
        ]

    def test_zone_can_silence(self):
        zones = [zone(["ppe"], left=True, notify=False)]
        _, report = run([ppe([violator(LEFT_BOX)], notify=True)], zones)
        assert report.notifications == []

    def test_one_detectors_zone_doesnt_speak_for_another(self):
        """A loud plate zone must not make a quiet PPE switch notify, and vice versa."""
        zones = [
            zone(["anpr"], left=False, notify=True, name="Entry", id=1),
            zone(["ppe"], left=True, notify=False, name="Yard", id=2),
        ]
        _, report = run(
            [ppe([violator(LEFT_BOX)], notify=True), anpr([plate("ABC")])], zones
        )
        assert [n.topic for n in report.notifications] == ["anpr_event"]


class TestEvents:
    def test_events_carry_detector_and_kind(self):
        _, report = run([ppe([violator(LEFT_BOX)]), anpr([plate("ABC")])])
        assert report.events == [
            ("ppe", {"kind": "ppe_violation", "count": 1, "missing": ["hard_hat"]}),
            ("anpr", {"kind": "anpr", "plate": "ABC", "confidence": 0.8}),
        ]

    def test_zone_filtered_findings_raise_no_event(self):
        _, report = run([ppe([violator(RIGHT_BOX)])], [zone(["ppe"], left=True)])
        assert report.events == []


class TestSeveralViews:
    def test_report_merges_views(self):
        """The processor reports once per message, across all its views."""
        a = pipeline.analyse([ppe([violator(LEFT_BOX)])], IMAGE, 640)
        b = pipeline.analyse([ppe([violator(RIGHT_BOX)])], IMAGE, 640)
        report = pipeline.report([ppe()], [a, b], "cam")
        assert report.summary == "2 people missing hard hat"
        assert len(report.notifications) == 1


class TestMatchToEvent:
    def test_classified_reasons_gate_specialists(self):
        p, a = ppe(), anpr()
        assert wanted_for_reason(p, "person")
        assert not wanted_for_reason(p, "vehicle")
        assert wanted_for_reason(a, "vehicle")
        assert not wanted_for_reason(a, "person")

    def test_unclassified_reasons_run_everything(self):
        for reason in ("schedule", "manual", "intruder", None):
            assert wanted_for_reason(ppe(), reason)
            assert wanted_for_reason(anpr(), reason)

    def test_detector_without_reasons_runs_on_anything(self):
        general = SimpleNamespace(camera_reasons=None)
        assert wanted_for_reason(general, "vehicle")
        assert wanted_for_reason(general, "person")


class TestLoadEnabled:
    def test_skips_disabled_missing_and_unset_sections(self):
        class Unset:
            @property
            def value(self):
                raise ValueError("not set")

        config = SimpleNamespace(
            ppe=SimpleNamespace(enabled=v(False)),
            # anpr absent entirely
            objects=SimpleNamespace(enabled=Unset()),
        )
        assert load_enabled(config) == []
