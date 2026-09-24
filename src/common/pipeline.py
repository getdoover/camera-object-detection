"""Run the detectors over a frame and decide what's worth saying about it.

This is the part both app shells used to carry their own copy of — run each model, filter
its findings by zone, summarise, pick the notifications — and so the part most likely to
drift between device and cloud. It lives here once, over the ``base.Detector`` contract,
so neither shell names a detector.

Two steps, because the processor analyses several views of one message and reports on
them together, where the device reports per frame:

* :func:`analyse` — one frame. Runs the models and applies the zones.
* :func:`report` — one or more analyses. Summary, ``camera_event`` payloads, and the
  notifications that survive the zone check.
"""

import logging
from dataclasses import dataclass, field

from . import annotate as annotate_mod
from . import zones as zones_mod
from .detectors.base import Detector, Notification

log = logging.getLogger(__name__)


def _box_of(item):
    return getattr(item, "box", None)


@dataclass
class Analysis:
    """One frame's results.

    ``results`` is what the models saw, unfiltered — it backs the annotated image and the
    published ``findings``. ``kept`` is what survived the zones, as ``(item, zone)`` pairs
    per detector, and is all that the report looks at.
    """

    results: dict = field(default_factory=dict)
    kept: dict = field(default_factory=dict)

    @property
    def findings(self) -> dict:
        return {name: result.to_dict() for name, result in self.results.items()}

    @property
    def found_anything(self) -> bool:
        return any(self.kept.values()) or any(
            r.has_findings for r in self.results.values()
        )

    def annotate(self, image):
        # Drawn in reverse run order so the specialists (PPE, plates) land on top: with a
        # `person` rule and PPE both on, the verdict is the box worth reading.
        return annotate_mod.annotate(image, reversed(list(self.results.values())))


def analyse(detectors: list[Detector], image, size: int, zones=None) -> Analysis:
    """Run every detector over ``image`` and filter each by its own zones.

    Blocking and CPU-bound — callers push it to a thread. A detector that raises is
    logged and left out rather than taking the others down with it.

    Each detector is filtered only by zones naming it: a PPE zone says nothing about where
    plates matter. Zones that name nothing we run are ignored, and no zones at all means
    "whole frame" (see ``zones.zones_for_detector``).
    """
    height, width = image.shape[:2]
    analysis = Analysis()

    for detector in detectors:
        try:
            result = detector.analyse(image, size)
        except Exception as e:
            log.error(f"{detector.name} inference failed: {e}", exc_info=e)
            continue
        analysis.results[detector.name] = result

        detector_zones = zones_mod.zones_for_detector(zones, detector.name)
        kept, dropped = zones_mod.filter_by_zones(
            result.reportable(), detector_zones, _box_of, width, height
        )
        analysis.kept[detector.name] = kept
        if dropped:
            # Said out loud, because a zone filtering everything out looks identical to a
            # detector that has stopped working.
            log.info(
                f"Ignoring {len(dropped)} {detector.name} finding(s) outside its "
                f"{len(detector_zones)} zone(s)."
            )
    return analysis


@dataclass
class Report:
    summary: str
    # (detector name, payload) — the payload has a `kind`; the shell adds who/when.
    events: list = field(default_factory=list)
    notifications: list = field(default_factory=list)
    # Flat numbers for tag history: counts and confidences per detector that ran, zeros
    # included. See each detector's `metrics`.
    metrics: dict = field(default_factory=dict)


def report(detectors: list[Detector], analyses: list[Analysis], camera: str) -> Report:
    """Summarise the zone-filtered findings across ``analyses``.

    ``camera`` is the display name, for text a person reads.

    A zone's ``notify`` overrides the detector's own switch in both directions, and only
    the zones that the alert's own findings fell in get a say — so one zone's setting
    can't silence, or trigger, another detector's alert.
    """
    parts, events, notifications, metrics = [], [], [], {}

    for detector in detectors:
        results = [
            a.results[detector.name] for a in analyses if detector.name in a.results
        ]
        if not results:
            # It failed on every frame (already logged). Recording zeros would claim
            # it looked and saw nothing.
            continue
        pairs = [p for a in analyses for p in a.kept.get(detector.name, [])]
        items = [item for item, _zone in pairs]
        metrics.update(detector.metrics(results, items))
        if not pairs:
            continue
        zone_of = {id(item): zone for item, zone in pairs}

        summary = detector.summary(items)
        if summary:
            parts.append(summary)

        events.extend((detector.name, e) for e in detector.events(items))

        for alert in detector.alerts(camera, items):
            scope = alert.items if alert.items is not None else items
            matched = [zone_of.get(id(item)) for item in scope]
            if zones_mod.should_notify(matched, alert.default_notify):
                notifications.append(
                    Notification(
                        f"{alert.text}{zone_suffix(matched)}.",
                        alert.severity,
                        alert.topic,
                    )
                )

    return Report(
        "; ".join(parts) or "nothing detected", events, notifications, metrics
    )


def zone_suffix(matched_zones) -> str:
    """ " in <zone>" when exactly one zone is responsible, else nothing.

    Left off when several zones are involved rather than listing them: the message is a
    headline, and the per-finding detail is already in the published payload.
    """
    named = {z.label for z in matched_zones if z is not None}
    if len(named) != 1:
        return ""
    return f" in {named.pop()}"
