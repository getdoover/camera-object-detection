"""Tests for rule-driven general object detection.

Most of these fake the model: what matters is how rules read its output — per-rule
confidence, counts, and that counting happens *after* the zones have filtered. One test
runs the real weights, because the class list here is duplicated from them.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from common import pipeline
from common.detectors.objects import (
    COCO_CLASSES,
    OBJECTS_MODEL_PATH,
    ObjectsDetector,
    Rule,
)
from common.yolo import Detection

from object_detection.app_config import ObjectDetectionConfig
from object_detection_processor.app_config import ObjectDetectionProcessorConfig

IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)
LEFT = (10, 10, 50, 90)
RIGHT = (150, 10, 190, 90)


def det(label, conf=0.9, box=LEFT):
    return Detection(label, conf, box)


class FakeModel:
    def __init__(self, detections):
        self.detections = detections
        self.class_names = dict(enumerate(COCO_CLASSES))
        self.calls = []

    def detect(self, image, confidence, size, wanted):
        self.calls.append({"confidence": confidence, "wanted": wanted})
        return [
            d
            for d in self.detections
            if d.confidence >= confidence and d.label in wanted
        ]


def detector(rules, detections=()):
    d = ObjectsDetector.__new__(ObjectsDetector)
    d.rules = rules
    d.model = FakeModel(list(detections))
    return d


def run(d, zones=None, camera="Yard"):
    analysis = pipeline.analyse([d], IMAGE, 640, zones)
    return analysis, pipeline.report([d], [analysis], camera)


class TestRule:
    def test_matches_class_and_confidence(self):
        rule = Rule("Cattle", ["cow"], confidence=60)
        assert rule.matches(det("cow", 0.6))
        assert not rule.matches(det("cow", 0.59))
        assert not rule.matches(det("horse", 0.9))

    def test_min_count(self):
        rule = Rule("Crowd", ["person"], min_count=3)
        assert rule.triggered_by([det("person")] * 2) == []
        assert len(rule.triggered_by([det("person")] * 3)) == 3

    def test_min_count_floor_is_one(self):
        """0 would mean 'trigger on an empty frame' — every snapshot a finding."""
        assert Rule("x", ["cow"], min_count=0).min_count == 1

    def test_from_config_names_an_unnamed_rule_by_its_objects(self):
        element = SimpleNamespace(
            name=SimpleNamespace(value=""),
            objects=SimpleNamespace(
                elements=[SimpleNamespace(value="cow"), SimpleNamespace(value="horse")]
            ),
            min_count=SimpleNamespace(value=2),
            confidence=SimpleNamespace(value=40),
            notify=SimpleNamespace(value=True),
        )
        rule = Rule.from_config(element)
        assert rule.name == "cow, horse"
        assert rule.classes == {"cow", "horse"}
        assert (rule.min_count, rule.confidence, rule.notify) == (2, 0.4, True)


class TestAnalyse:
    def test_one_model_run_at_the_loosest_threshold(self):
        d = detector(
            [
                Rule("Cattle", ["cow"], confidence=70),
                Rule("Dogs", ["dog"], confidence=40),
            ]
        )
        d.analyse(IMAGE, 640)
        assert d.model.calls == [{"confidence": 0.4, "wanted": {"cow", "dog"}}]

    def test_keeps_only_what_a_rule_wants_at_its_own_threshold(self):
        d = detector(
            [
                Rule("Cattle", ["cow"], confidence=70),
                Rule("Dogs", ["dog"], confidence=40),
            ],
            [det("cow", 0.5), det("cow", 0.8), det("dog", 0.5), det("chair", 0.9)],
        )
        result = d.analyse(IMAGE, 640)
        assert [(x.label, x.confidence) for x in result.detections] == [
            ("cow", 0.8),
            ("dog", 0.5),
        ]

    def test_no_rules_skips_the_model(self):
        d = detector([], [det("cow")])
        assert not d.analyse(IMAGE, 640).has_findings
        assert d.model.calls == []


class TestReporting:
    def test_summary_event_and_alert(self):
        d = detector(
            [Rule("Cattle in laneway", ["cow", "horse"], notify=True)],
            [det("cow"), det("cow"), det("horse")],
        )
        analysis, report = run(d)
        assert analysis.findings["objects"]["objects"][0]["label"] == "cow"
        assert report.summary == "Cattle in laneway: 2 x cow, 1 x horse"
        assert report.events == [
            (
                "objects",
                {
                    "kind": "object_detected",
                    "rule": "Cattle in laneway",
                    "count": 3,
                    "objects": ["cow", "horse"],
                },
            )
        ]
        assert [n.text for n in report.notifications] == [
            "Cattle in laneway: Yard saw 2 x cow, 1 x horse."
        ]

    def test_below_min_count_reports_nothing_but_is_still_seen(self):
        d = detector([Rule("Crowd", ["person"], min_count=3)], [det("person")] * 2)
        analysis, report = run(d)
        assert analysis.found_anything
        assert report.summary == "nothing detected"
        assert report.events == []

    def test_rules_are_independent(self):
        d = detector(
            [Rule("Cattle", ["cow"]), Rule("Crowd", ["person"], min_count=5)],
            [det("cow"), det("person")],
        )
        _, report = run(d)
        assert report.summary == "Cattle: 1 x cow"

    def test_notify_off_by_default(self):
        d = detector([Rule("Cattle", ["cow"])], [det("cow")])
        _, report = run(d)
        assert report.notifications == []


class TestZones:
    def zone(self, notify=True, name="Laneway"):
        return {
            "id": 1,
            "kind": "intrusion",
            "detectors": ["objects"],
            "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "notify": notify,
            "name": name,
        }

    def test_count_is_taken_after_the_zone_filter(self):
        """Two cows wanted in the laneway: one in it and one in the paddock isn't that."""
        d = detector(
            [Rule("Cattle", ["cow"], min_count=2)],
            [det("cow", box=LEFT), det("cow", box=RIGHT)],
        )
        _, report = run(d, [self.zone()])
        assert report.summary == "nothing detected"

    def test_zone_notify_overrides_rule_and_names_itself(self):
        d = detector([Rule("Cattle", ["cow"], notify=False)], [det("cow", box=LEFT)])
        _, report = run(d, [self.zone(notify=True)])
        assert [n.text for n in report.notifications] == [
            "Cattle: Yard saw 1 x cow in Laneway."
        ]


def existing_install(schema) -> dict:
    """A real config saved before the objects section existed (the simulator's)."""
    data = json.loads(
        (Path(__file__).parents[1] / "simulators" / "app_config.json").read_text()
    )
    if schema is ObjectDetectionProcessorConfig:
        data.pop("camera_app")
        data["dv_proc_subscriptions"] = "doover_camera_1"
    return data


# Keyed as a saved config is: by display name, not by Python attribute.
RULE = {
    "name": "Cattle",
    "objects": ["cow"],
    "minimum_count": 2,
    "minimum_confidence": 60,
    "notify": True,
}


class TestConfig:
    @pytest.mark.parametrize(
        "schema", [ObjectDetectionConfig, ObjectDetectionProcessorConfig]
    )
    def test_install_without_the_section_still_loads_as_off(self, schema):
        """Every install saved before this detector existed has no `objects` key."""
        data = existing_install(schema)
        assert "object_detection" not in data
        config = schema()
        config._inject_deployment_config(data)
        assert config.objects.enabled.value is False
        assert config.objects.rules.elements == []

    @pytest.mark.parametrize(
        "schema", [ObjectDetectionConfig, ObjectDetectionProcessorConfig]
    )
    def test_rules_load(self, schema):
        config = schema()
        data = existing_install(schema)
        data["object_detection"] = {
            "enabled": True,
            "rules": [RULE, {"objects": ["dog"]}],
        }
        config._inject_deployment_config(data)
        rules = [Rule.from_config(e) for e in config.objects.rules.elements]
        assert [(r.name, r.classes, r.min_count, r.confidence) for r in rules] == [
            ("Cattle", {"cow"}, 2, 0.6),
            ("dog", {"dog"}, 1, 0.5),
        ]

    def test_object_picker_offers_the_coco_vocabulary(self):
        schema = ObjectDetectionConfig.to_schema()
        rule = schema["properties"]["object_detection"]["properties"]["rules"]
        choices = rule["items"]["properties"]["objects"]["items"]["enum"]
        assert choices == list(COCO_CLASSES)


@pytest.mark.skipif(not OBJECTS_MODEL_PATH.exists(), reason="weights not fetched")
def test_real_model_vocabulary_and_a_real_frame():
    """The class list above is a copy of the weights'; this is what keeps it honest."""
    d = ObjectsDetector(
        SimpleNamespace(rules=SimpleNamespace(elements=[])),
        model_path=OBJECTS_MODEL_PATH,
    )
    assert tuple(d.model.class_names[i] for i in range(80)) == COCO_CLASSES

    ultralytics = pytest.importorskip("ultralytics")
    import cv2

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    d.rules = [Rule("People", ["person"]), Rule("Buses", ["bus"])]
    analysis = pipeline.analyse([d], image, 640)
    report = pipeline.report([d], [analysis], "cam")
    assert report.summary == "People: 3 x person; Buses: 1 x bus"
