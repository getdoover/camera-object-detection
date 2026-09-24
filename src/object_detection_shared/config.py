"""Config shared by the device app and the cloud processor.

Each install analyses **one camera**, so everything here is that camera's settings: which
detectors run, their thresholds, and the object rules. The two variants subclass
:class:`DetectionConfig` and add only how they're triggered — the device names the
camera app to subscribe to, the processor takes a channel subscription.

Kept outside ``common`` because it imports pydoover, and ``common`` is deliberately free
of that. Kept outside both app packages because the processor can't import
``object_detection``: its ``__init__`` pulls in the device runtime, which needs grpc, and
the Lambda image doesn't ship it.

The stored config keys come from the display names ("Minimum Confidence" ->
``minimum_confidence``), not the attribute names, so renaming a display name renames the
key.
"""

from common.detectors.objects import COCO_CLASSES
from pydoover import config

# Values the camera app puts in a snapshot's `reason`.
SNAPSHOT_REASONS = [
    "schedule",
    "manual",
    "intruder",
    "person",
    "vehicle",
    "anpr",
    "ppe",
]


class PPEConfig(config.Object):
    """Hard-hat / high-vis compliance. See ``common/detectors/ppe.py``."""

    enabled = config.Boolean(
        "Enabled",
        description="Check people for hard hats and high-vis vests.",
        default=False,
    )
    require_hard_hat = config.Boolean(
        "Require Hard Hat",
        description="Flag a person who isn't wearing a hard hat.",
        default=True,
    )
    require_high_vis = config.Boolean(
        "Require High-Vis",
        description="Flag a person who isn't wearing a high-vis vest.",
        default=True,
    )
    # 55 is measured, not guessed: on a live 4K yard frame a traffic cone came back as
    # `person` 0.49 and was flagged for no hard hat. But a genuine person on an awkward
    # wide-angle frame scored 0.50, so this is a trade, not a fix — see the README.
    confidence = config.Integer(
        "Minimum Confidence",
        description="Ignore detections below this confidence (0-100). Lower finds more "
        "distant people but starts reading orange cones and plant as people in hi-vis.",
        default=55,
        minimum=1,
        maximum=100,
    )
    notify_on_violation = config.Boolean(
        "Notify On Violation",
        description="Send a notification when someone is missing required PPE.",
        default=True,
    )


class ANPRConfig(config.Object):
    """Number-plate detection + OCR. See ``common/detectors/anpr.py``."""

    enabled = config.Boolean(
        "Enabled",
        description="Find and read vehicle number plates.",
        default=False,
    )
    confidence = config.Integer(
        "Minimum Confidence",
        description="Ignore plate detections below this confidence (0-100).",
        default=40,
        minimum=1,
        maximum=100,
    )
    min_plate_chars = config.Integer(
        "Minimum Plate Characters",
        description="Discard reads shorter than this. Guards against a partial read "
        "being reported as a real plate.",
        default=4,
        minimum=1,
        maximum=12,
        advanced=True,
    )
    notify_on_plate = config.Boolean(
        "Notify On Plate Read",
        description="Send a notification for every plate read. Can be a lot on a busy "
        "gate.",
        default=False,
    )


class ObjectRuleConfig(config.Object):
    """One rule: "tell me when at least N of these objects are in frame"."""

    name = config.String(
        "Name",
        description="What to call this in the timeline, tags and notifications, e.g. "
        "'Cattle in laneway'.",
        default="",
    )
    objects = config.Array(
        "Objects",
        description="Any of these counts towards the rule.",
        element=config.Enum("Object", choices=list(COCO_CLASSES), default="person"),
        default=[],
    )
    min_count = config.Integer(
        "Minimum Count",
        description="How many must be in frame (and in an 'objects' zone, if the camera "
        "has any) for the rule to trigger.",
        default=1,
        minimum=1,
        maximum=100,
    )
    confidence = config.Integer(
        "Minimum Confidence",
        description="Ignore detections below this confidence (0-100).",
        default=50,
        minimum=1,
        maximum=100,
    )
    notify = config.Boolean(
        "Notify",
        description="Send a notification when this rule triggers.",
        default=False,
    )


class ObjectsConfig(config.Object):
    """General-purpose COCO detection. See ``common/detectors/objects.py``."""

    enabled = config.Boolean(
        "Enabled",
        description="Look for everyday objects (people, vehicles, animals, ...) using "
        "the rules below.",
        default=False,
    )
    rules = config.Array(
        "Rules",
        description="What to look for. The model runs once per frame however many "
        "rules there are.",
        element=ObjectRuleConfig("Rule"),
        default=[],
    )


class DetectionConfig(config.Schema):
    """Everything about analysing a camera, minus how snapshots arrive."""

    # Every section is defaulted to off, so a new install saves with nothing filled in
    # and each camera opts into only the detectors it needs.
    ppe = PPEConfig("PPE Detection", default={"enabled": False})
    anpr = ANPRConfig("Number Plate Recognition", default={"enabled": False})
    objects = ObjectsConfig("Object Detection", default={"enabled": False, "rules": []})

    analyse_reasons = config.Array(
        "Analyse Snapshots Because Of",
        description="Only analyse snapshots the camera took for these reasons. Empty "
        "means all of them. A camera's own 'Object Detection' snapshot setting "
        "overrides this either way.",
        element=config.Enum("Reason", choices=SNAPSHOT_REASONS, default="person"),
        default=[],
    )
    annotate = config.Boolean(
        "Annotate Images",
        description="Attach a copy of the frame with labelled boxes drawn on it.",
        default=True,
    )
    publish_clean_results = config.Boolean(
        "Publish Results With No Findings",
        description="Record an analysis even when nothing was seen. Off so the "
        "camera's timeline isn't filled with empty results.",
        default=False,
        advanced=True,
    )
    # Not an accuracy dial: the weights are trained at 640, and on a real site frame
    # 960 lost a person that 640 found. See the README.
    inference_size = config.Integer(
        "Inference Size",
        description="Square size (px) frames are resized to before inference. Leave at "
        "640 unless you've measured otherwise; bigger is not more accurate.",
        default=640,
        minimum=320,
        maximum=1280,
        advanced=True,
    )

    @property
    def wanted_reasons(self) -> set[str]:
        """Snapshot reasons to analyse. Empty set means "everything"."""
        return {e.value for e in self.analyse_reasons.elements if e.value}
