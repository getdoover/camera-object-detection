"""Tags shared by the device app and the cloud processor.

One install is one camera, so these are that camera's figures, flat. Two kinds:

* **Per analysis** — what the last analysed frame contained (counts, confidences, and a
  count per object rule). Written *and logged to history* on every analysis, including
  zeros and repeats, so the history is a record of each frame rather than of changes.
  A detector that isn't enabled leaves its tags alone.
* **Running** — counters and "last seen" values that dashboards read as current state.

Confidences are 0-1, matching ``findings`` and ``camera_event``; 0 means nothing was
seen (see ``common.detectors.base.confidence_stats``).
"""

from datetime import datetime, timezone

from common.detectors.objects import rules_from_config
from pydoover import tags


class ObjectDetectionTags(tags.Tags):
    # -- running ------------------------------------------------------------
    analysed_count = tags.Number(default=0)
    violation_count = tags.Number(default=0)
    # Epoch milliseconds, matching the camera app's tag of the same name so a dashboard
    # can read either interchangeably.
    last_ppe_violation = tags.Number(default=0)
    last_plate = tags.String(default="")

    # -- per analysis -------------------------------------------------------
    # Epoch milliseconds of the analysis the other per-analysis tags describe.
    last_analysed_at = tags.Number(default=0)

    ppe_people = tags.Number(default=0)
    ppe_violations = tags.Number(default=0)
    ppe_max_confidence = tags.Number(default=0)
    ppe_mean_confidence = tags.Number(default=0)

    anpr_plates = tags.Number(default=0)
    anpr_plates_read = tags.Number(default=0)
    anpr_max_confidence = tags.Number(default=0)
    anpr_mean_confidence = tags.Number(default=0)

    objects_count = tags.Number(default=0)
    objects_max_confidence = tags.Number(default=0)
    objects_mean_confidence = tags.Number(default=0)

    async def setup(self):
        # One tag per object rule (`rule_<name>`), from the same function the detector
        # names them with, so the declared set always matches what gets written.
        try:
            enabled = self.config.objects.enabled.value
        except ValueError:
            enabled = False
        if not enabled:
            return
        for rule in rules_from_config(self.config.objects):
            self.add_tag(rule.tag_name, tags.Number(default=0))


async def update_running_tags(bound: ObjectDetectionTags, events) -> None:
    """Advance the running tags from a report's ``(detector, event)`` pairs.

    These predate generic detection and dashboards read them, so they keep tracking PPE
    and plates specifically; object rules have their per-analysis counts instead.
    """
    for _detector, event in events:
        kind = event.get("kind")
        if kind == "anpr":
            await bound.last_plate.set(event["plate"])
        elif kind == "ppe_violation":
            await bound.violation_count.set(bound.violation_count.value + 1)
            # Epoch milliseconds, matching the camera app's tag of the same name. Its
            # naive `datetime.now()` yields the same epoch value as this, since
            # `timestamp()` reads a naive datetime as local time.
            await bound.last_ppe_violation.set(
                int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            )
