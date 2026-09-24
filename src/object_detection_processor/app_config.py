from pathlib import Path

from common.detectors.objects import COCO_CLASSES
from pydoover import config
from pydoover.processor import ManySubscriptionConfig


class PPEConfig(config.Object):
    """Hard-hat / high-vis compliance checking. See `common/detectors/ppe.py`."""

    enabled = config.Boolean(
        "Enabled",
        description="Run hard-hat / high-vis detection on incoming snapshots.",
        default=True,
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
    confidence = config.Integer(
        "Minimum Confidence",
        description="Drop detections below this confidence (0-100).",
        default=55,
        minimum=1,
        maximum=100,
        advanced=True,
    )
    notify_on_violation = config.Boolean(
        "Notify On Violation",
        description="Send a notification when someone is missing required PPE.",
        default=True,
    )


class ANPRConfig(config.Object):
    """Number-plate detection + OCR. See `common/detectors/anpr.py`."""

    enabled = config.Boolean(
        "Enabled",
        description="Detect and read vehicle number plates on incoming snapshots.",
        default=False,
    )
    confidence = config.Integer(
        "Minimum Confidence",
        description="Drop plate detections below this confidence (0-100).",
        default=40,
        minimum=1,
        maximum=100,
        advanced=True,
    )
    min_plate_chars = config.Integer(
        "Minimum Plate Characters",
        description="Discard OCR results shorter than this many characters.",
        default=4,
        minimum=1,
        maximum=12,
        advanced=True,
    )
    notify_on_plate = config.Boolean(
        "Notify On Plate Read",
        description="Send a notification for every plate read.",
        default=False,
    )


class ObjectRuleConfig(config.Object):
    """One rule: "tell me when at least N of these objects are in frame"."""

    name = config.String(
        "Name",
        description="What to call this in the timeline and notifications.",
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
        description="How many must be in frame (and in an 'objects' zone, if any).",
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
        advanced=True,
    )
    notify = config.Boolean(
        "Notify",
        description="Send a notification when this rule triggers.",
        default=False,
    )


class ObjectsConfig(config.Object):
    """General-purpose COCO detection. See `common/detectors/objects.py`."""

    enabled = config.Boolean(
        "Enabled",
        description="Run general object detection on incoming snapshots.",
        default=False,
    )
    rules = config.Array(
        "Rules",
        description="What to look for. Every rule reads the same single model run.",
        element=ObjectRuleConfig("Rule"),
        default=[],
    )


class ObjectDetectionProcessorConfig(config.Schema):
    """Config for the cloud processor.

    Deliberately has no camera-app list: a processor is invoked by a subscription, so
    the platform decides which channels reach it. The on-device app needs that list
    because it subscribes itself.
    """

    ppe = PPEConfig("PPE Detection")
    anpr = ANPRConfig("Number Plate Recognition")
    objects = ObjectsConfig("Object Detection", default={"enabled": False, "rules": []})

    analyse_reasons = config.Array(
        "Analyse Snapshots Because Of",
        description="Only analyse snapshots captured for these reasons. Matches the "
        "'reason' field the camera app publishes. Empty means analyse everything.",
        element=config.Enum(
            "Reason",
            description="Snapshot reason",
            choices=[
                "schedule",
                "manual",
                "intruder",
                "person",
                "vehicle",
                "anpr",
                "ppe",
            ],
            default="person",
        ),
    )

    annotate = config.Boolean(
        "Annotate Images",
        description="Attach a copy of the frame with labelled boxes drawn on it.",
        default=True,
    )
    match_detectors_to_event = config.Boolean(
        "Match Detectors To Event",
        description="Only run the detector the camera's classification calls for: PPE "
        "on a person event, plates on a vehicle event. Halves the inference per frame "
        "and removes false findings from the model that wasn't relevant (a traffic cone "
        "read as a person on a vehicle event, for instance). Unclassified snapshots "
        "(schedule, manual, intruder) still run everything, and general object "
        "detection runs on every event. Turn off to always run all of them -- worth "
        "doing if people arrive by vehicle and you need PPE checked on the vehicle "
        "event itself.",
        default=True,
        advanced=True,
    )
    inference_size = config.Integer(
        "Inference Size",
        description="Square size (px) frames are letterboxed to before inference. "
        "Leave at 640 unless you have measured otherwise: raising it is NOT a free "
        "accuracy win. The weights are trained at 640, and on a real site frame 960 "
        "lost a person that 640 found (see the README). More CPU here buys throughput, "
        "not better detection.",
        default=640,
        minimum=320,
        maximum=1920,
        advanced=True,
    )

    channels = ManySubscriptionConfig()

    @property
    def wanted_reasons(self) -> set:
        """Snapshot reasons to analyse. Empty set means "everything"."""
        return {e.value for e in self.analyse_reasons.elements if e.value}


def export():
    ObjectDetectionProcessorConfig().export(
        Path(__file__).parents[2] / "doover_config.json",
        "object_detection_processor",
    )


if __name__ == "__main__":
    export()
