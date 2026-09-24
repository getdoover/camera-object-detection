"""Tests for what each analysis records to tag history, and the config around it.

The history is only useful if every analysis lands in it — zeros and repeats included —
so the tests that matter most here are the ones pinning how the two shells write: the
device must bypass pydoover's skip-if-unchanged, and both must ask for a log.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from common import pipeline
from common.detectors.anpr import ANPRDetector, ANPRResult, Plate
from common.detectors.base import confidence_stats
from common.detectors.objects import (
    ObjectsDetector,
    ObjectsResult,
    Rule,
    rules_from_config,
)
from common.detectors.ppe import Person, PPEDetector, PPEResult
from common.yolo import Detection
from object_detection_shared.tags import ObjectDetectionTags, update_running_tags

from object_detection.app_config import ObjectDetectionConfig
from object_detection.application import ObjectDetectionApplication
from object_detection_processor.app_config import ObjectDetectionProcessorConfig
from object_detection_processor.application import ObjectDetectionProcessor

IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)
LEFT = (10, 10, 50, 90)
RIGHT = (150, 10, 190, 90)


def v(value):
    return SimpleNamespace(value=value)


def rule_config(*rules):
    elements = [
        SimpleNamespace(
            name=v(name),
            objects=SimpleNamespace(elements=[v(c) for c in classes]),
            min_count=v(1),
            confidence=v(50),
            notify=v(False),
        )
        for name, classes in rules
    ]
    return SimpleNamespace(rules=SimpleNamespace(elements=elements))


class Canned:
    """A detector whose `analyse` returns a fixed result; reporting is the real one's."""

    def __init__(self, real, result):
        self._real, self._result = real, result
        self.name, self.camera_reasons = real.name, real.camera_reasons

    def analyse(self, image, size):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def __getattr__(self, attr):
        return getattr(self._real, attr)


def ppe(people):
    real = PPEDetector.__new__(PPEDetector)
    real.config = SimpleNamespace(notify_on_violation=v(False))
    result = PPEResult(people, [])
    result.violators = [p for p in people if p.missing]
    return Canned(real, result)


def person(box, conf, missing=()):
    p = Person(Detection("person", conf, box))
    p.missing = list(missing)
    return p


def anpr(plates):
    real = ANPRDetector.__new__(ANPRDetector)
    real.config = SimpleNamespace(notify_on_plate=v(False))
    return Canned(real, ANPRResult(plates))


def objects(rules, detections):
    real = ObjectsDetector.__new__(ObjectsDetector)
    real.rules = rules
    return Canned(real, ObjectsResult(detections))


def zone(detectors, left=True):
    x0, x1 = (0.0, 0.5) if left else (0.5, 1.0)
    return {
        "id": 1,
        "kind": "intrusion",
        "detectors": detectors,
        "points": [[x0, 0.0], [x1, 0.0], [x1, 1.0], [x0, 1.0]],
    }


def metrics(detectors, zones=None):
    analysis = pipeline.analyse(detectors, IMAGE, 640, zones)
    return pipeline.report(detectors, [analysis], "cam").metrics


class TestConfidenceStats:
    def test_max_and_mean(self):
        items = [Detection("cow", 0.9, LEFT), Detection("cow", 0.6, LEFT)]
        assert confidence_stats("objects", items) == {
            "objects_max_confidence": 0.9,
            "objects_mean_confidence": 0.75,
        }

    def test_nothing_seen_is_zero_not_unset(self):
        """None would clear the tag, leaving a hole indistinguishable from an outage."""
        assert confidence_stats("ppe", []) == {
            "ppe_max_confidence": 0,
            "ppe_mean_confidence": 0,
        }

    def test_reads_through_wrapped_detections(self):
        assert confidence_stats("ppe", [person(LEFT, 0.8)])["ppe_max_confidence"] == 0.8


class TestMetrics:
    def test_ppe_counts_seen_people_and_zone_filtered_violations(self):
        people = [person(LEFT, 0.9, ["hard_hat"]), person(RIGHT, 0.7, ["hard_hat"])]
        got = metrics([ppe(people)], [zone(["ppe"], left=True)])
        assert got == {
            "ppe_people": 2,
            "ppe_violations": 1,
            "ppe_max_confidence": 0.9,
            "ppe_mean_confidence": 0.8,
        }

    def test_anpr_counts_unread_plates_as_seen_but_not_read(self):
        plates = [
            Plate(Detection("plate", 0.8, LEFT), "ABC123"),
            Plate(Detection("plate", 0.4, RIGHT), None),
        ]
        got = metrics([anpr(plates)])
        assert got["anpr_plates"] == 2
        assert got["anpr_plates_read"] == 1
        assert got["anpr_max_confidence"] == 0.8

    def test_objects_rule_counts_are_after_zones_and_below_min_count_too(self):
        cattle = Rule("Cattle", ["cow"], min_count=3)
        dogs = Rule("Dogs", ["dog"])
        dets = [
            Detection("cow", 0.9, LEFT),
            Detection("cow", 0.8, LEFT),
            Detection("cow", 0.7, RIGHT),
            Detection("dog", 0.6, RIGHT),
        ]
        got = metrics([objects([cattle, dogs], dets)], [zone(["objects"], left=True)])
        assert got["objects_count"] == 4
        # Two cows in the zone: short of the rule's 3, but still the count.
        assert got["rule_cattle"] == 2
        assert got["rule_dogs"] == 0

    def test_empty_frame_records_zeros(self):
        got = metrics([ppe([]), anpr([])])
        assert got["ppe_people"] == 0
        assert got["anpr_plates"] == 0
        assert got["ppe_max_confidence"] == 0

    def test_failed_detector_records_nothing(self):
        """Zeros would claim it looked and saw nothing."""
        broken = ppe([])
        broken._result = RuntimeError("boom")
        got = metrics([broken, anpr([])])
        assert "ppe_people" not in got
        assert got["anpr_plates"] == 0

    def test_every_metric_has_a_declared_tag(self):
        """A metric without a declaration is written but invisible to the tag schema."""
        declared = set(ObjectDetectionTags.__tag_declarations__)
        got = metrics([ppe([]), anpr([]), objects([], [])])
        assert set(got) <= declared


class TestRuleTagNames:
    def test_slugged_and_unique(self):
        rules = rules_from_config(
            rule_config(("Cattle in laneway!", ["cow"]), ("cattle in LANEWAY", ["cow"]))
        )
        assert [r.tag_name for r in rules] == [
            "rule_cattle_in_laneway",
            "rule_cattle_in_laneway_2",
        ]

    def test_rule_with_no_objects_is_dropped(self):
        rules = rules_from_config(rule_config(("Empty", []), ("Dogs", ["dog"])))
        assert [r.tag_name for r in rules] == ["rule_dogs"]

    def test_unnamed_rule_is_named_by_its_objects(self):
        assert rules_from_config(rule_config(("", ["cow", "horse"])))[0].tag_name == (
            "rule_cow_horse"
        )


def load(schema, **objects_section):
    data = json.loads(
        (Path(__file__).parents[1] / "simulators" / "app_config.json").read_text()
    )
    if schema is ObjectDetectionProcessorConfig:
        data.pop("camera_app")
        data["dv_proc_subscriptions"] = "doover_camera_1"
    if objects_section:
        data["object_detection"] = objects_section
    config = schema()
    config._inject_deployment_config(data)
    return config


class TestTagDeclarations:
    def test_one_tag_per_rule_when_objects_enabled(self):
        config = load(
            ObjectDetectionConfig,
            enabled=True,
            rules=[{"name": "Cattle", "objects": ["cow"]}, {"objects": ["dog"]}],
        )
        tags = ObjectDetectionTags("od_1", None, config)
        asyncio.run(tags.setup())
        assert {"rule_cattle", "rule_dog"} <= set(tags._tag_declarations)

    def test_no_rule_tags_when_objects_disabled(self):
        config = load(
            ObjectDetectionConfig,
            enabled=False,
            rules=[{"name": "Cattle", "objects": ["cow"]}],
        )
        tags = ObjectDetectionTags("od_1", None, config)
        asyncio.run(tags.setup())
        assert "rule_cattle" not in tags._tag_declarations


class FakeTag:
    def __init__(self, value):
        self.value = value

    async def set(self, value, log=False):
        self.value = value


class TestRunningTags:
    def test_plates_and_violations(self):
        bound = SimpleNamespace(
            last_plate=FakeTag(""),
            violation_count=FakeTag(2),
            last_ppe_violation=FakeTag(0),
        )
        events = [
            ("anpr", {"kind": "anpr", "plate": "ABC123", "confidence": 0.9}),
            ("ppe", {"kind": "ppe_violation", "count": 1, "missing": ["hard_hat"]}),
            ("objects", {"kind": "object_detected", "rule": "x", "count": 1}),
        ]
        asyncio.run(update_running_tags(bound, events))
        assert bound.last_plate.value == "ABC123"
        assert bound.violation_count.value == 3
        assert bound.last_ppe_violation.value > 0


class TestShellsLogEveryAnalysis:
    def test_device_bypasses_skip_if_unchanged_and_logs(self):
        calls = []

        class Manager:
            async def set_tags(self, tags, only_if_changed=True, log=False, **kw):
                calls.append((tags, only_if_changed, log))

        app = ObjectDetectionApplication.__new__(ObjectDetectionApplication)
        app.app_key = "od_1"
        app.tag_manager = Manager()
        asyncio.run(app._record_metrics({"ppe_people": 2}))

        ((tags, only_if_changed, log),) = calls
        assert only_if_changed is False
        assert log is True
        assert tags["od_1"]["ppe_people"] == 2
        assert tags["od_1"]["last_analysed_at"] > 0

    def test_processor_logs_each_value(self):
        calls = []

        class Manager:
            async def set_tag(self, key, value, log=False, **kw):
                calls.append((key, value, log))

        app = ObjectDetectionProcessor.__new__(ObjectDetectionProcessor)
        app.tag_manager = Manager()
        app.tags = SimpleNamespace(
            analysed_count=FakeTag(4),
            last_plate=FakeTag(""),
            violation_count=FakeTag(0),
            last_ppe_violation=FakeTag(0),
        )
        report = pipeline.Report("x", events=[], metrics={"objects_count": 3})
        asyncio.run(app._record_tags(report))

        assert app.tags.analysed_count.value == 5
        logged = {k: (val, log) for k, val, log in calls}
        assert logged["objects_count"] == (3, True)
        assert logged["last_analysed_at"][1] is True


class TestConfigShape:
    @pytest.mark.parametrize(
        "schema, key",
        [
            (ObjectDetectionConfig, "camera_app"),
            (ObjectDetectionProcessorConfig, "dv_proc_subscriptions"),
        ],
    )
    def test_one_camera_defaulting_to_the_first(self, schema, key):
        prop = schema.to_schema()["properties"][key]
        assert prop["type"] in ("string", ["string", "null"])
        assert prop["default"] == "doover_camera_1"
        # First in the form.
        positions = [
            p.get("x-position", 0) for p in schema.to_schema()["properties"].values()
        ]
        assert prop["x-position"] == min(positions)

    @pytest.mark.parametrize(
        "schema", [ObjectDetectionConfig, ObjectDetectionProcessorConfig]
    )
    def test_confidence_thresholds_are_not_hidden(self, schema):
        props = schema.to_schema()["properties"]
        rule = props["object_detection"]["properties"]["rules"]["items"]
        for section in (
            props["ppe_detection"],
            props["number_plate_recognition"],
            rule,
        ):
            conf = section["properties"]["minimum_confidence"]
            assert not conf.get("x-advanced"), section["title"]

    def test_variants_share_the_detection_settings(self):
        device = ObjectDetectionConfig.to_schema()["properties"]
        cloud = ObjectDetectionProcessorConfig.to_schema()["properties"]
        shared = set(device) & set(cloud)
        assert {
            "ppe_detection",
            "number_plate_recognition",
            "object_detection",
        } <= shared
        for key in ("ppe_detection", "number_plate_recognition", "object_detection"):
            assert device[key] == cloud[key]


class TestNotifications:
    """Alerts name a declared notification; the declaration sets topic and severity."""

    def test_every_alert_event_is_declared(self):
        from object_detection_shared.notifications import ObjectDetectionNotifications

        declared = set(ObjectDetectionNotifications().to_schema())
        events = {"ppe_violation", "plate_read", "object_rule"}
        assert events <= declared
        # And the detectors really do use those names.
        real_ppe = PPEDetector.__new__(PPEDetector)
        real_ppe.config = SimpleNamespace(notify_on_violation=v(True))
        real_anpr = ANPRDetector.__new__(ANPRDetector)
        real_anpr.config = SimpleNamespace(notify_on_plate=v(True))
        real_obj = ObjectsDetector.__new__(ObjectsDetector)
        real_obj.rules = [Rule("Cattle", ["cow"], notify=True)]
        alerted = {
            *(
                a.event
                for a in real_ppe.alerts("cam", [person(LEFT, 0.9, ["hard_hat"])])
            ),
            *(
                a.event
                for a in real_anpr.alerts(
                    "cam", [Plate(Detection("plate", 0.9, LEFT), "ABC")]
                )
            ),
            *(a.event for a in real_obj.alerts("cam", [Detection("cow", 0.9, LEFT)])),
        }
        assert alerted == events

    @pytest.mark.parametrize(
        "event, severity",
        [("ppe_violation", "Warn"), ("plate_read", "Info"), ("object_rule", "Info")],
    )
    def test_topic_and_severity(self, event, severity):
        """The topic is a contract with the API and the subscription editor."""
        from object_detection_shared.notifications import ObjectDetectionNotifications
        from pydoover.models import NotificationSeverity

        sent = []

        class App:
            async def send_notification(self, message, **kwargs):
                sent.append((message, kwargs))

        bound = ObjectDetectionNotifications("object_detection_1", App())
        asyncio.run(bound[event].send("Yard cam saw something."))

        ((message, kwargs),) = sent
        assert message == "Yard cam saw something."
        assert str(kwargs["topic"]) == (
            f"dev/applications/default/object_detection_1/{event}"
        )
        assert kwargs["severity"] is getattr(NotificationSeverity, severity)
        # No title: the server substitutes the agent's display name.
        assert kwargs["title"] is None
