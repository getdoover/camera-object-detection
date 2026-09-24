"""General-purpose object detection, driven by user-defined rules.

One COCO-class model (80 everyday classes — people, vehicles, animals, bags, ...) runs
once per frame, and any number of **rules** read its output: "cattle in the laneway" is
`cow`, at least 1; "crowding" is `person`, at least 6. Adding a rule costs nothing at
inference time, which is the point — on a CM4 the price is per *model* (~1.4s a frame),
not per class, so one general model beats a model per object type.

What this is not good at is *absence of an attribute* ("person without a hard hat"). A
general model has no notion of it, which is why PPE and plates stay as their own
detectors with their own reasoning.

Weights: Ultralytics YOLO11n trained on COCO, exported to ONNX by
``scripts/fetch_models.py``. **AGPL-3.0** — see the licence note in the README.
"""

import logging
import re

from ..yolo import MODEL_DIR, Detection, ModelUnavailable, YoloOnnx
from .base import (
    SEVERITY_INFO,
    STYLE_OBJECT,
    Alert,
    Annotation,
    confidence_stats,
    describe_labels,
)

log = logging.getLogger(__name__)

OBJECTS_MODEL_PATH = MODEL_DIR / "objects.onnx"

# The model's vocabulary, in class-index order. Duplicated from the weights on purpose:
# the config schema's object picker is exported at build time, long before any model is
# loaded. The detector checks the loaded model against this at startup, so a swapped-in
# model with a different vocabulary is caught loudly rather than silently never matching.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip


class Rule:
    """ "Tell me when at least ``min_count`` of these objects are in frame"."""

    def __init__(
        self,
        name: str,
        classes,
        min_count: int = 1,
        confidence: int = 50,
        notify: bool = False,
    ):
        self.name = name
        self.classes = frozenset(c.lower() for c in classes)
        self.min_count = max(1, int(min_count))
        self.confidence = confidence / 100
        self.notify = notify
        # The tag this rule's count is recorded under. Assigned by `rules_from_config`,
        # which is the one place that can make it unique across rules.
        self.tag_name = f"rule_{_slug(name)}"

    @classmethod
    def from_config(cls, element) -> "Rule":
        classes = [e.value for e in element.objects.elements if e.value]
        return cls(
            # A rule with no name still needs something a person can read in the
            # timeline and a notification; its objects are the obvious stand-in.
            name=element.name.value or ", ".join(classes) or "unnamed rule",
            classes=classes,
            min_count=element.min_count.value,
            confidence=element.confidence.value,
            notify=element.notify.value,
        )

    def matches(self, detection: Detection) -> bool:
        return (
            detection.label in self.classes and detection.confidence >= self.confidence
        )

    def matching(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if self.matches(d)]

    def triggered_by(self, detections: list[Detection]) -> list[Detection]:
        """The detections that trigger this rule, or [] if there aren't enough."""
        matched = self.matching(detections)
        return matched if len(matched) >= self.min_count else []


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"


def rules_from_config(config) -> list[Rule]:
    """The usable rules in an objects config section, with unique tag names.

    Shared by the detector and the tag declarations, so the tags an install declares are
    always exactly the ones the detector writes. A rule that names no objects can never
    match anything, so it's dropped (loudly) rather than declared as a tag that's
    always 0.
    """
    rules = []
    for element in config.rules.elements:
        rule = Rule.from_config(element)
        if not rule.classes:
            log.warning(f"Ignoring rule '{rule.name}': it names no objects.")
            continue
        rules.append(rule)

    # Two rules can slug to the same name ("Cows!" and "cows"); number the repeats so
    # neither overwrites the other's history.
    seen: dict[str, int] = {}
    for rule in rules:
        n = seen.get(rule.tag_name, 0) + 1
        seen[rule.tag_name] = n
        if n > 1:
            rule.tag_name = f"{rule.tag_name}_{n}"
    return rules


class ObjectsResult:
    def __init__(self, detections: list[Detection]):
        # Only what some rule asked about. A COCO model finds chairs, cups and potted
        # plants in most frames; drawing and publishing those would bury the thing the
        # rule is for.
        self.detections = detections

    @property
    def has_findings(self) -> bool:
        return bool(self.detections)

    def reportable(self) -> list[Detection]:
        # Every relevant detection goes forward to the zone filter. Whether a rule
        # *triggers* is decided after that, on what's left: a rule wanting two cows in
        # the laneway must not fire on one cow in the laneway and one in the paddock.
        return list(self.detections)

    def annotations(self) -> list[Annotation]:
        return [Annotation(d.box, STYLE_OBJECT, d.label) for d in self.detections]

    def to_dict(self) -> dict:
        return {"objects": [d.to_dict() for d in self.detections]}


class ObjectsDetector:
    name = "objects"
    # Runs on any camera event. The camera only classifies people and vehicles, so its
    # classification says nothing about whether a cow or a dog is in frame.
    camera_reasons = None

    def __init__(self, config, model_path=OBJECTS_MODEL_PATH):
        self.rules = rules_from_config(config)
        self.model = YoloOnnx(model_path)

        available = set(self.model.class_names.values())
        if available and available != set(COCO_CLASSES):
            log.warning(
                f"{model_path.name} doesn't have the COCO vocabulary the config offers "
                f"({len(available)} classes). Rules naming a class it lacks will never "
                f"trigger."
            )
        for rule in self.rules:
            missing = rule.classes - available
            if missing:
                log.error(
                    f"Rule '{rule.name}' asks for {sorted(missing)}, which "
                    f"{model_path.name} can't detect."
                )
        if not self.rules:
            log.warning(
                "Object detection is enabled but has no rules with any objects in them, "
                "so it will never report anything."
            )

    def analyse(self, image, size: int) -> ObjectsResult:
        """Run the model once and keep what any rule wants. CPU-bound; use a thread."""
        if not self.rules:
            return ObjectsResult([])

        detections = self.model.detect(
            image,
            # The loosest rule's threshold, so one pass serves every rule; each rule
            # then applies its own.
            confidence=min(r.confidence for r in self.rules),
            size=size,
            wanted=set().union(*(r.classes for r in self.rules)),
        )
        return ObjectsResult(
            [d for d in detections if any(r.matches(d) for r in self.rules)]
        )

    # -- reporting (detections here are already zone-filtered) ----------------

    def triggered(self, detections: list[Detection]) -> list[tuple[Rule, list]]:
        out = []
        for rule in self.rules:
            matched = rule.triggered_by(detections)
            if matched:
                out.append((rule, matched))
        return out

    def summary(self, detections: list[Detection]) -> str | None:
        parts = [
            f"{rule.name}: {describe_labels(matched)}"
            for rule, matched in self.triggered(detections)
        ]
        return "; ".join(parts) or None

    def events(self, detections: list[Detection]) -> list[dict]:
        return [
            {
                "kind": "object_detected",
                "rule": rule.name,
                "count": len(matched),
                "objects": sorted({d.label for d in matched}),
            }
            for rule, matched in self.triggered(detections)
        ]

    def metrics(self, results: list[ObjectsResult], detections: list) -> dict:
        """Numbers for tag history: what was seen, and each rule's count after zones.

        A rule's count is recorded whether or not it reached its minimum, so the history
        shows "2 cows" on the way to a rule that wants 3.
        """
        seen = [d for r in results for d in r.detections]
        return {
            "objects_count": len(seen),
            **confidence_stats("objects", seen),
            **{rule.tag_name: len(rule.matching(detections)) for rule in self.rules},
        }

    def alerts(self, camera: str, detections: list[Detection]) -> list[Alert]:
        return [
            Alert(
                f"{rule.name}: {camera} saw {describe_labels(matched)}",
                SEVERITY_INFO,
                topic="object_event",
                default_notify=rule.notify,
                # This rule's own detections decide which zones get a say.
                items=matched,
            )
            for rule, matched in self.triggered(detections)
        ]


def load(config) -> ObjectsDetector | None:
    try:
        return ObjectsDetector(config)
    except ModelUnavailable as e:
        log.error(
            f"Object detection is enabled but the model can't be loaded: {e}. Run "
            f"scripts/fetch_models.py and rebuild the image."
        )
        return None
