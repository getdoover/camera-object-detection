from pydoover.tags import Tag, Tags

from .app_config import ObjectDetectionConfig


class ObjectDetectionTags(Tags):
    config: ObjectDetectionConfig

    analysed_count = Tag("number", 0)
    violation_count = Tag("number", 0)
    last_plate = Tag("string", "")
    # Epoch milliseconds, matching the camera app's tag of the same name so a
    # dashboard can read either interchangeably.
    last_ppe_violation = Tag("number", 0)
