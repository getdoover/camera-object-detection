"""Hard-hat / high-vis compliance from a PPE object-detection model.

The model detects *equipment*, not compliance. It emits independent boxes for
`person`, `hardhat`, `no-hardhat`, `safety vest` and `no-safety vest`, and it is
this module's job to decide what that means per person.

Two things make that non-trivial:

* The `no-*` classes are unreliable on their own. A model will happily emit both a
  `hardhat` and a `no-hardhat` box over the same head at similar confidence. So a
  person is treated as compliant if a *positive* box matches them, and only flagged
  when no positive box does -- the `no-*` boxes are used as corroboration and to
  catch people the `person` class missed, never as the sole basis for a violation.
* Equipment boxes have to be attributed to a person. IoU is the wrong measure: a
  hard hat is a tiny box inside a large person box, so their IoU is near zero even
  when they obviously belong together. What matters is how much of the *equipment*
  box falls inside the person box, which is what :func:`_containment` measures.
"""

import logging

from ..yolo import MODEL_DIR, Detection, ModelUnavailable, YoloOnnx
from .base import (
    SEVERITY_WARN,
    STYLE_BAD,
    STYLE_OK,
    Alert,
    Annotation,
    confidence_stats,
)

log = logging.getLogger(__name__)

PPE_MODEL_PATH = MODEL_DIR / "ppe.onnx"

# The PPE model's class names -> what they mean to us. Model vocabularies differ
# between published PPE weights (hyphens vs spaces, "helmet" vs "hardhat"), so
# every spelling we've seen is mapped rather than assuming one.
HARD_HAT_PRESENT = {"hardhat", "hard-hat", "helmet"}
HARD_HAT_MISSING = {"no-hardhat", "no hardhat", "no-hard-hat", "no helmet", "nohelmet"}
HIGH_VIS_PRESENT = {"safety vest", "safety-vest", "vest", "hi-vis", "high-vis"}
HIGH_VIS_MISSING = {
    "no-safety vest",
    "no-safety-vest",
    "no safety vest",
    "no-vest",
    "no vest",
}
PERSON = {"person"}

# Fraction of the equipment box that must fall inside a person box to count as worn
# by them. Hats sit at the very top of a person box and vests are fully enclosed, so
# a genuine match is nearly total; 0.5 leaves room for a slightly loose person box
# without letting a hat on the ground next to someone count as worn.
CONTAINMENT_THRESHOLD = 0.5

# When the model finds no `person` at all but does find a bare head / vestless torso,
# that equipment box is treated as standing in for a person. Its box is a fraction of
# a body, so it's grown into something person-shaped purely so the annotated image
# draws a sensible box around who was flagged.
IMPLIED_PERSON_SCALE = 3.0


def _containment(inner: Detection, outer: Detection) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1, ix2, iy2 = inner.box
    ox1, oy1, ox2, oy2 = outer.box

    ox = max(0, min(ix2, ox2) - max(ix1, ox1))
    oy = max(0, min(iy2, oy2) - max(iy1, oy1))
    overlap = ox * oy
    return overlap / inner.area if inner.area else 0.0


class Person:
    """A detected person and the PPE attributed to them."""

    def __init__(self, detection: Detection, implied: bool = False):
        self.detection = detection
        # None = the model said nothing either way about this person.
        self.hard_hat: bool = None
        self.high_vis: bool = None
        # True when this "person" was inferred from a bare-head/vestless box because
        # the person class missed them.
        self.implied = implied

    @property
    def box(self):
        return self.detection.box

    def violations(self, require_hard_hat: bool, require_high_vis: bool) -> list[str]:
        """Which requirements this person fails.

        ``None`` (the model didn't say) counts as a failure only for a requirement
        that's switched on -- if we're asked to police hard hats and can't see one on
        somebody, that's the thing worth flagging.
        """
        missing = []
        if require_hard_hat and not self.hard_hat:
            missing.append("hard_hat")
        if require_high_vis and not self.high_vis:
            missing.append("high_vis")
        return missing

    def to_dict(self) -> dict:
        return {
            "box": list(self.detection.box),
            "confidence": round(self.detection.confidence, 3),
            "hard_hat": self.hard_hat,
            "high_vis": self.high_vis,
            "implied": self.implied,
        }


_MISSING_LABELS = {"hard_hat": "NO HARD HAT", "high_vis": "NO HI-VIS"}


def _pretty_missing(violators, joiner: str) -> str:
    missing = sorted({m for v in violators for m in v.missing})
    return joiner.join(m.replace("_", " ") for m in missing)


class PPEResult:
    def __init__(self, people: list[Person], raw: list[Detection]):
        self.people = people
        self.raw = raw
        self.violators: list[Person] = []

    @property
    def has_findings(self) -> bool:
        return bool(self.people)

    def reportable(self) -> list[Person]:
        # Compliant people are drawn and published, but there's nothing to report.
        return list(self.violators)

    def annotations(self) -> list[Annotation]:
        out = []
        for person in self.people:
            missing = getattr(person, "missing", None) or []
            if missing:
                label = " + ".join(_MISSING_LABELS.get(m, m.upper()) for m in missing)
                out.append(Annotation(person.box, STYLE_BAD, label))
            else:
                out.append(Annotation(person.box, STYLE_OK, "PPE OK"))
        return out

    def to_dict(self) -> dict:
        return {
            "people": [p.to_dict() for p in self.people],
            "violations": [
                {"box": list(p.detection.box), "missing": p.missing}
                for p in self.violators
            ],
        }


class PPEDetector:
    name = "ppe"
    # PPE is about people, so a vehicle event is not a reason to run it: on one, this
    # model returned a traffic cone as a person at 0.49 and flagged it for no hard hat.
    camera_reasons = frozenset({"person", "ppe"})

    def __init__(self, config):
        self.config = config
        self.model = YoloOnnx(PPE_MODEL_PATH)

        available = set(self.model.class_names.values())
        # Fail loudly at startup rather than silently reporting "nobody in shot"
        # forever because the weights use a vocabulary we don't map.
        if not available & PERSON:
            log.warning(
                f"{PPE_MODEL_PATH.name} has no 'person' class ({sorted(available)}); "
                f"people will only be inferred from bare-head/vestless boxes."
            )
        if self.config.require_hard_hat.value and not available & (
            HARD_HAT_PRESENT | HARD_HAT_MISSING
        ):
            log.error(
                f"Hard-hat checking is on but {PPE_MODEL_PATH.name} has no hard-hat "
                f"class ({sorted(available)}). It will never flag anything."
            )
        if self.config.require_high_vis.value and not available & (
            HIGH_VIS_PRESENT | HIGH_VIS_MISSING
        ):
            log.error(
                f"High-vis checking is on but {PPE_MODEL_PATH.name} has no vest class "
                f"({sorted(available)}). It will never flag anything."
            )

    def analyse(self, image, size: int) -> PPEResult:
        """Detect people and attribute PPE to them. CPU-bound; call in a thread."""
        wanted = (
            PERSON
            | HARD_HAT_PRESENT
            | HARD_HAT_MISSING
            | HIGH_VIS_PRESENT
            | HIGH_VIS_MISSING
        )
        detections = self.model.detect(
            image,
            confidence=self.config.confidence.value / 100,
            size=size,
            wanted=wanted,
        )

        people = [Person(d) for d in detections if d.label in PERSON]
        equipment = [d for d in detections if d.label not in PERSON]

        unclaimed = []
        for item in equipment:
            if not self._assign(item, people):
                unclaimed.append(item)

        # Anyone the person class missed but whose bare head / vestless torso was
        # detected still needs flagging -- that's exactly the case we care about.
        for item in unclaimed:
            if item.label in HARD_HAT_MISSING or item.label in HIGH_VIS_MISSING:
                person = self._imply_person(item, image.shape[:2])
                if item.label in HARD_HAT_MISSING:
                    person.hard_hat = False
                else:
                    person.high_vis = False
                people.append(person)

        result = PPEResult(people, detections)
        for person in people:
            person.missing = person.violations(
                self.config.require_hard_hat.value,
                self.config.require_high_vis.value,
            )
            if person.missing:
                result.violators.append(person)
        return result

    # -- reporting ------------------------------------------------------------
    #
    # `violators` here are already zone-filtered, which is why these take a list rather
    # than reading PPEResult.violators.

    def summary(self, violators: list[Person]) -> str | None:
        if not violators:
            return None
        who = "person" if len(violators) == 1 else "people"
        return f"{len(violators)} {who} missing {_pretty_missing(violators, ', ')}"

    def events(self, violators: list[Person]) -> list[dict]:
        if not violators:
            return []
        return [
            {
                "kind": "ppe_violation",
                "count": len(violators),
                "missing": sorted({m for v in violators for m in v.missing}),
            }
        ]

    def metrics(self, results: list[PPEResult], violators: list[Person]) -> dict:
        """People seen, and violations after zones. Confidence is of the people."""
        people = [p for r in results for p in r.people]
        return {
            "ppe_people": len(people),
            "ppe_violations": len(violators),
            **confidence_stats("ppe", people),
        }

    def alerts(self, camera: str, violators: list[Person]) -> list[Alert]:
        if not violators:
            return []
        who = "someone" if len(violators) == 1 else f"{len(violators)} people"
        return [
            Alert(
                f"{camera} detected {who} without {_pretty_missing(violators, ' and ')}",
                SEVERITY_WARN,
                topic="ppe_event",
                default_notify=self.config.notify_on_violation.value,
            )
        ]

    @staticmethod
    def _assign(item: Detection, people: list[Person]) -> bool:
        """Attribute an equipment box to whichever person most encloses it."""
        best, best_score = None, CONTAINMENT_THRESHOLD
        for person in people:
            score = _containment(item, person.detection)
            if score >= best_score:
                best, best_score = person, score
        if best is None:
            return False

        # A positive detection always wins over a negative one for the same person:
        # these models routinely emit both `hardhat` and `no-hardhat` over one head,
        # and treating that as a violation would be a false alarm.
        if item.label in HARD_HAT_PRESENT:
            best.hard_hat = True
        elif item.label in HARD_HAT_MISSING and best.hard_hat is None:
            best.hard_hat = False
        elif item.label in HIGH_VIS_PRESENT:
            best.high_vis = True
        elif item.label in HIGH_VIS_MISSING and best.high_vis is None:
            best.high_vis = False
        return True

    @staticmethod
    def _imply_person(item: Detection, shape: tuple[int, int]) -> Person:
        """Grow an equipment box into a person-shaped box for annotation."""
        h, w = shape
        x1, y1, x2, y2 = item.box
        bw, bh = x2 - x1, y2 - y1
        grow_w = int(bw * (IMPLIED_PERSON_SCALE - 1) / 2)
        grow_h = int(bh * (IMPLIED_PERSON_SCALE - 1))
        box = (
            max(0, x1 - grow_w),
            max(0, y1),
            min(w, x2 + grow_w),
            # A bare head implies a body *below* it, so grow downward only.
            min(h, y2 + grow_h),
        )
        return Person(Detection("person", item.confidence, box), implied=True)


def load(config) -> PPEDetector | None:
    """Build the detector, or return None with a clear reason if it can't run."""
    try:
        return PPEDetector(config)
    except ModelUnavailable as e:
        log.error(
            f"PPE detection is enabled but the model can't be loaded: {e}. Run "
            f"scripts/fetch_models.py and rebuild the image."
        )
        return None
