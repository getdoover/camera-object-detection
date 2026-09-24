from pathlib import Path

from common.detectors.objects import COCO_CLASSES
from pydoover import config


class PPEConfig(config.Object):
    """Hard-hat / high-vis compliance checking.

    The model classifies *equipment*, not compliance: it emits `person`,
    `hardhat`/`no-hardhat` and `safety vest`/`no-safety vest` boxes independently.
    Turning that into "this person is missing a hard hat" is done in
    detectors/ppe.py by matching equipment boxes onto person boxes, which is why
    the two requirements below are separate switches rather than one "PPE" flag.
    """

    enabled = config.Boolean(
        "Enabled",
        description="Run hard-hat / high-vis detection on incoming snapshots.",
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
    confidence = config.Integer(
        "Minimum Confidence",
        description="Drop detections below this confidence (0-100). Lower catches more "
        "distant people but invents them: on a real site camera a traffic cone came "
        "back as a person at 0.49, which at the old default of 40 raised a violation.",
        # Measured, not guessed. On a live 4K frame from an AcuSense pointed at a yard,
        # an orange cone on a wall produced `person` 0.49 + `safety vest` 0.69 -- a
        # phantom person flagged for no hard hat.
        #
        # 55 suppresses that, but it is a trade, not a fix: a *genuine* person on a
        # similarly awkward wide-angle frame scored 0.50, so the two are not separable
        # by confidence and this default also loses that detection. Properly framed
        # subjects score 0.83-0.84, far clear of the noise -- which is why the README
        # says the camera angle matters more than this number. Don't lower it to chase
        # a badly-framed camera without expecting cones and orange plant to flag.
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
    """Number-plate detection + OCR.

    Two models: a YOLO plate *detector* crops the plate out of the frame, then an
    OCR model reads the crop. A plate that's detected but unreadable still counts
    as a detection (and is drawn on the annotated image) with no text.
    """

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
        description="Discard OCR results shorter than this many characters. Guards "
        "against a partial read being published as a real plate.",
        default=4,
        minimum=1,
        maximum=12,
        advanced=True,
    )
    notify_on_plate = config.Boolean(
        "Notify On Plate Read",
        description="Send a notification for every plate read. Off by default -- on a "
        "busy site this is a lot of notifications.",
        default=False,
    )


class ObjectRuleConfig(config.Object):
    """One rule: "tell me when at least N of these objects are in frame"."""

    name = config.String(
        "Name",
        description="What to call this in the timeline and notifications, e.g. "
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
        description="How many of them must be in frame (and in a zone, if the camera has "
        "'objects' zones) for the rule to trigger.",
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
    """General-purpose detection with a COCO model (80 everyday object classes).

    The model runs once per frame whatever the number of rules, so rules are free; see
    common/detectors/objects.py.
    """

    enabled = config.Boolean(
        "Enabled",
        description="Run general object detection on incoming snapshots. Costs about as "
        "much CPU per frame as the PPE model, regardless of how many rules there are.",
        default=False,
    )
    rules = config.Array(
        "Rules",
        description="What to look for. Every rule reads the same single model run.",
        element=ObjectRuleConfig("Rule"),
        default=[],
    )


class ObjectDetectionConfig(config.Schema):
    camera_app_keys = config.Array(
        "Camera Apps",
        description="App keys of the camera apps whose snapshots to analyse, e.g. "
        "'doover_camera_1'. One instance can watch several cameras -- preferred over "
        "deploying one instance per camera, since each instance loads its own copy of "
        "the models into RAM.",
        element=config.String("App Key", description="Camera app key"),
    )

    ppe = PPEConfig("PPE Detection")
    anpr = ANPRConfig("Number Plate Recognition")
    # Defaulted, so an install saved before this section existed still loads (as off).
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
            default="intruder",
        ),
    )

    annotate = config.Boolean(
        "Annotate Images",
        description="Draw labelled boxes on the frame and publish it back to the "
        "camera's channel so the timeline shows what was flagged.",
        default=True,
    )
    publish_clean_results = config.Boolean(
        "Publish Results With No Findings",
        description="Publish a result even when nothing was detected. Off by default so "
        "the camera timeline isn't filled with empty analyses.",
        default=False,
        advanced=True,
    )
    inference_size = config.Integer(
        "Inference Size",
        description="Square size (px) frames are letterboxed to before inference. "
        "Larger catches smaller/more distant subjects but costs CPU time and RAM.",
        default=640,
        minimum=320,
        maximum=1280,
        advanced=True,
    )

    @property
    def watched_app_keys(self) -> list[str]:
        return [e.value for e in self.camera_app_keys.elements if e.value]

    @property
    def wanted_reasons(self) -> set[str]:
        """Snapshot reasons to analyse. Empty set means "everything"."""
        return {e.value for e in self.analyse_reasons.elements if e.value}


def export():
    ObjectDetectionConfig().export(
        Path(__file__).parents[2] / "doover_config.json", "object_detection"
    )


if __name__ == "__main__":
    export()
