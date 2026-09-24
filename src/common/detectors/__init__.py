"""The detectors, and the one place that knows which exist.

Adding a detector means writing a module that satisfies ``base.Detector``, adding its
loader here, and giving both app configs a section of the same name. Nothing in the app
shells or ``common.pipeline`` names a detector.
"""

import logging

from . import anpr, objects, ppe
from .anpr import ANPRDetector, ANPRResult, Plate
from .base import CLASSIFIED_REASONS, Detector, wanted_for_reason
from .objects import COCO_CLASSES, ObjectsDetector, ObjectsResult, Rule
from .ppe import Person, PPEDetector, PPEResult

log = logging.getLogger(__name__)

# Name -> loader. The name is the config section, the ``findings`` key and the zone
# detector key all at once, so it must not change once deployed.
#
# Order is the order they run and are summarised in.
LOADERS = {
    "ppe": ppe.load,
    "anpr": anpr.load,
    "objects": objects.load,
}


def _enabled(section) -> bool:
    try:
        return bool(section.enabled.value)
    except ValueError:
        # A section the deployment has never saved (an install older than the
        # detector) can arrive with nothing set. That means off, not a crash at start.
        return False


def load_enabled(config) -> list[Detector]:
    """Build every detector whose config section is enabled and whose weights load.

    ``config`` is either app's schema; each detector reads the attribute of its own name.
    A detector that fails to load has already logged why and is left out.
    """
    detectors = []
    for name, loader in LOADERS.items():
        section = getattr(config, name, None)
        if section is None or not _enabled(section):
            continue
        detector = loader(section)
        if detector is not None:
            detectors.append(detector)
    return detectors


__all__ = (
    "CLASSIFIED_REASONS",
    "COCO_CLASSES",
    "LOADERS",
    "ANPRDetector",
    "ANPRResult",
    "Detector",
    "ObjectsDetector",
    "ObjectsResult",
    "PPEDetector",
    "PPEResult",
    "Person",
    "Plate",
    "Rule",
    "anpr",
    "load_enabled",
    "objects",
    "ppe",
    "wanted_for_reason",
)
