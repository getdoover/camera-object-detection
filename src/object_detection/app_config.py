from pathlib import Path

from object_detection_shared.config import DetectionConfig
from pydoover import config


class ObjectDetectionConfig(DetectionConfig):
    """One install per camera. The detection settings come from ``DetectionConfig``."""

    # Listed first in the form (the shared elements are numbered from 0): it's the one
    # thing that has to be set before anything happens.
    camera_app = config.Application(
        "Camera App",
        description="The camera app whose snapshots to analyse. Install this app once "
        "per camera.",
        default="doover_camera_1",
        position=-1,
    )

    @property
    def camera_app_key(self) -> str | None:
        return self.camera_app.value or None


def export():
    ObjectDetectionConfig().export(
        Path(__file__).parents[2] / "doover_config.json", "object_detection"
    )


if __name__ == "__main__":
    export()
