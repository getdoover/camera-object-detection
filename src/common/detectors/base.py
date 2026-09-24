"""The contract every detector implements, so the app shells don't know any by name.

A detector is two halves, and the split is what zones need:

* **Seeing** — :meth:`Detector.analyse` runs the model(s) over a frame and returns a
  result. The result describes *everything* the models saw (``to_dict``, and what gets
  drawn), and names the subset worth reporting (``reportable``): PPE violators, plates
  that were actually read, objects a rule is interested in.
* **Reporting** — the detector turns reportable items into a summary, ``camera_event``
  payloads, notifications and tag-history metrics. These take items *after* the zone filter has run, which is
  why they live on the detector rather than the result: the shells (via
  ``common.pipeline``) sit between the two halves and narrow the list.

Nothing here imports doover. Events are plain dicts, and an alert names the declared
notification it's for (``object_detection_shared.notifications``) rather than carrying a
topic or severity, so the device app and the Lambda processor each send it through their
own platform calls.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

# Annotation styles. annotate.py owns the colours; detectors only say what a box means.
STYLE_BAD = "bad"
STYLE_OK = "ok"
STYLE_PLATE = "plate"
STYLE_OBJECT = "object"


@dataclass
class Annotation:
    box: tuple[int, int, int, int]
    style: str
    label: str = ""


@dataclass
class Alert:
    """A notification a detector would like sent, before zones have had their say.

    ``text`` has no trailing full stop: the pipeline appends " in <zone>" when one zone is
    responsible. ``event`` is the name of the declared notification to send it as, which
    sets its topic and severity. ``default_notify`` is this detector's own config switch, which a matching
    zone overrides in either direction (see ``zones.should_notify``).

    ``items`` narrows which findings' zones decide this alert. None means all of the
    detector's findings, which is right when one alert covers them all (PPE); a detector
    raising one alert per rule passes that rule's items, so an unrelated finding's zone
    can't speak for it.
    """

    text: str
    event: str
    default_notify: bool
    items: list | None = None


@dataclass
class Notification:
    """An alert that survived the zone check, ready to send."""

    text: str
    event: str


class Result(Protocol):
    @property
    def has_findings(self) -> bool:
        """Whether the models saw anything at all, reportable or not.

        Drives "Publish Results With No Findings": a compliant worker or an unread plate
        is still worth a timeline entry even though nothing is reported.
        """

    def to_dict(self) -> dict:
        """Everything seen, unfiltered. Published under ``findings[<detector name>]``."""

    def reportable(self) -> list:
        """The items worth reporting. Each must have a ``box`` (pixels) or None."""

    def annotations(self) -> list[Annotation]: ...


class Detector(Protocol):
    # The key under ``findings``, and what a zone names in its ``detectors`` list.
    name: str

    # The camera classifications (snapshot ``reason``s) this detector is relevant to, or
    # None for "any". Used by the processor's "Match Detectors To Event" switch — see
    # :func:`wanted_for_reason`.
    camera_reasons: frozenset[str] | None

    def analyse(self, image, size: int) -> Result:
        """Run the models. Blocking and CPU-bound; callers push it to a thread."""

    def summary(self, items: list) -> str | None: ...

    def events(self, items: list) -> list[dict]:
        """``camera_event`` payloads. Each has a ``kind``; the shell adds the rest."""

    def alerts(self, camera: str, items: list) -> list[Alert]: ...

    def metrics(self, results: list, items: list) -> dict:
        """Numbers for tag history, prefixed with the detector's name.

        ``results`` are every frame's results for this detector (unfiltered: what was
        *seen*); ``items`` are the reportable items that survived the zones. Recorded on
        every analysis, zeros included — "nothing here" is a data point too.
        """


# Snapshot reasons that carry the camera's own classification of what it saw. Only these
# can rule a detector out; any other reason (schedule, manual, intruder) says nothing
# about what's in frame, so everything runs.
#
# Fixed here rather than derived from the enabled detectors' `camera_reasons`: with only
# PPE enabled, "vehicle" would then look unclassified and run PPE over a vehicle event —
# exactly the traffic-cone-as-person false positive this gating exists to prevent.
CLASSIFIED_REASONS = frozenset({"person", "ppe", "vehicle", "anpr"})


def wanted_for_reason(detector: Detector, reason: str | None) -> bool:
    """Whether the camera's classification of a snapshot justifies running ``detector``."""
    if reason not in CLASSIFIED_REASONS or detector.camera_reasons is None:
        return True
    return reason in detector.camera_reasons


def describe_labels(items: list[Any]) -> str:
    """ "3 x cow, 1 x horse" — most frequent first, then alphabetical."""
    counts = Counter(item.label for item in items)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{n} x {label}" for label, n in ordered)


def confidence_stats(prefix: str, items: list) -> dict:
    """``<prefix>_max_confidence`` / ``<prefix>_mean_confidence``, 0-1.

    0 when nothing was seen, rather than unset: a tag set to None is cleared, and a
    history with holes in it can't be told apart from an app that stopped reporting.
    """
    values = [_confidence_of(item) for item in items]
    values = [v for v in values if v is not None]
    if not values:
        return {f"{prefix}_max_confidence": 0, f"{prefix}_mean_confidence": 0}
    return {
        f"{prefix}_max_confidence": round(max(values), 3),
        f"{prefix}_mean_confidence": round(sum(values) / len(values), 3),
    }


def _confidence_of(item):
    detection = getattr(item, "detection", item)
    return getattr(detection, "confidence", None)
