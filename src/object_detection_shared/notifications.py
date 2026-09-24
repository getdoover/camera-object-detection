"""Notifications both variants can send, declared so each has its own topic.

Declaring them publishes a schema with the app, which is what lets an operator turn one
off (say, plate reads) without losing the rest. The attribute names are the event names
in the topic (``dev/applications/default/<app_key>/<event>``), and the detectors name
them in their alerts (``common.detectors.base.Alert.event``), so renaming one here
changes its topic and orphans any subscriber's setting for the old one.

Object rules share one notification rather than one each: rules are per-install config,
and the schema is exported from this class at publish time, before any install exists.
The rule's name leads the message instead.

Every message is overridden on send with the specifics (who, what, which zone); the
message here is the fallback and what the site shows in its picker. No title: the server
substitutes the agent's display name.
"""

from pathlib import Path

from pydoover import notifications


class ObjectDetectionNotifications(notifications.Notifications):
    ppe_violation = notifications.Notification(
        "Someone is missing required PPE",
        display_name="PPE violation",
        description="Sent when a person in a snapshot is missing a required hard hat "
        "or high-vis vest.",
        severity=notifications.NotificationSeverity.Warn,
    )
    plate_read = notifications.Notification(
        "A number plate was read",
        display_name="Number plate read",
        description="Sent for every plate read in a snapshot. Can be a lot on a busy "
        "gate.",
        severity=notifications.NotificationSeverity.Info,
    )
    object_rule = notifications.Notification(
        "An object detection rule triggered",
        display_name="Object rule triggered",
        description="Sent when one of this camera's object rules triggers, e.g. "
        "'Cattle in laneway'. The message names the rule.",
        severity=notifications.NotificationSeverity.Info,
    )


def _export(app_name: str):
    ObjectDetectionNotifications.export(
        Path(__file__).parents[2] / "doover_config.json", app_name
    )


def export():
    _export("object_detection")


def export_processor():
    _export("object_detection_processor")
