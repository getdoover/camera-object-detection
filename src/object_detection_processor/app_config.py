from pathlib import Path

from object_detection_shared.config import DetectionConfig
from pydoover import config
from pydoover.processor import SubscriptionConfig


class ObjectDetectionProcessorConfig(DetectionConfig):
    """One install per camera. The detection settings come from ``DetectionConfig``."""

    # A camera app publishes its snapshots on a channel named after its app key, so
    # subscribing to that one channel is what makes this install that camera's.
    # First in the form, for the same reason as the device app's Camera App.
    channel = SubscriptionConfig(
        "Camera Channel",
        description="The camera's snapshot channel: its app key, e.g. 'doover_camera_1'.",
        default="doover_camera_1",
        position=-1,
    )

    match_detectors_to_event = config.Boolean(
        "Match Detectors To Event",
        description="Skip a detector the camera's own classification rules out: PPE "
        "only on person events, plates only on vehicle events. Removes false findings "
        "such as a traffic cone read as a person on a vehicle event. Snapshots with no "
        "classification (schedule, manual, intruder) run everything, and object rules "
        "run on every event.",
        default=True,
        advanced=True,
    )


def export():
    ObjectDetectionProcessorConfig().export(
        Path(__file__).parents[2] / "doover_config.json",
        "object_detection_processor",
    )


if __name__ == "__main__":
    export()
