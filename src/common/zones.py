"""Detection zones as a spatial filter over what the models found.

The camera app owns the zone editor and hands the zones that concern this app along with
each snapshot (its ``detection_zones`` payload key). A zone names what it wants in its
``detectors`` list — any detector's ``name`` (``ppe``, ``anpr``, ``objects``), in any
combination — independently of its ``kind``, which describes how the *camera* triggers
it and is none of our business.

Why a filter and not a crop: the models run over the whole frame regardless. Cropping to
the zone first would be cheaper, but a person standing half out of the zone would be cut in
two and a plate at the boundary lost outright, and the models are much worse on partial
subjects. So the frame is analysed whole and each finding is then asked which zone it fell
in — which also means one inference run serves any number of zones.

Zone geometry is normalised (0..1, origin top-left) because that is what the frontend
drew and what the camera app stores; model detections are in **original image pixels**
(see ``yolo.Detection``). Bridging those two is this module's job and the reason
:func:`match` needs the frame size.

Deliberately free of doover coupling, like everything else in ``common``: plain dicts and
tuples in, plain objects out, so the device app and the Lambda processor share it verbatim.
"""

# What a zone can ask us to look for: the `name` of each detector (see
# `detectors.LOADERS`). A zone names these in its `detectors` list and can carry several —
# one polygon wanting a person's hard hat and any plate in the same frame.
#
# KNOWN_DETECTORS is only the legacy set below, not every detector: these are NOT zone
# kinds. `kind` describes how the *camera* triggers the zone
# (`intrusion` / `excluded_area`) and is none of our business; an earlier camera app made
# PPE and ANPR kinds of their own, which is why `zones_for_detector` still honours that
# spelling.
DETECTOR_PPE = "ppe"
DETECTOR_ANPR = "anpr"
DETECTOR_OBJECTS = "objects"
KNOWN_DETECTORS = (DETECTOR_PPE, DETECTOR_ANPR)


class Zone:
    """One zone as the camera app describes it.

    Mirrors ``camera_app.events.DetectionZone`` (in the getdoover/cameras repo) over the
    wire. It is not imported from there — these are two separately deployed apps in
    separate repos with no shared package — so the field names here are a contract, and
    changing one without the other breaks the filter silently. The round trip is covered
    by tests on both sides.
    """

    def __init__(self, kind, points, notify=True, name=None, id=None, detectors=None):
        # How the camera triggers this zone. Carried for logging only — what we look for
        # is `detectors`, and a zone of either kind can ask for either detector.
        self.kind = kind
        self.points = points
        self.notify = notify
        self.name = name
        self.id = id
        self.detectors = detectors or []

    @property
    def label(self) -> str:
        return self.name or (f"zone {self.id}" if self.id is not None else "zone")

    @classmethod
    def from_dict(cls, payload: dict) -> "Zone | None":
        """Build a zone, or None if the payload isn't a usable one.

        A zone needs at least three points to enclose anything; anything less is dropped
        rather than treated as matching nothing, so a malformed entry can't quietly
        swallow every detection in the frame.
        """
        if not isinstance(payload, dict):
            return None

        points = []
        for point in payload.get("points") or []:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError, IndexError):
                return None
        if len(points) < 3:
            return None

        detectors = payload.get("detectors")
        if not isinstance(detectors, list):
            # An older camera app made PPE/ANPR zone *kinds* rather than detectors. Read
            # the kind as a detector so a zone drawn against that app keeps being
            # analysed instead of silently dropping out of scope.
            kind = payload.get("kind")
            detectors = [kind] if kind in KNOWN_DETECTORS else []

        return cls(
            kind=payload.get("kind"),
            detectors=[d for d in detectors if isinstance(d, str)],
            points=points,
            # Absent means notify. This app's own config switch is the other half of the
            # decision (see the callers), and a zone that didn't state a preference must
            # not be the thing that silences it.
            notify=bool(payload.get("notify", True)),
            name=payload.get("name"),
            id=payload.get("id"),
        )

    def contains(self, x: float, y: float) -> bool:
        """Whether a normalised point is inside the polygon (ray casting).

        Handles the concave shapes the editor allows, which a bounding-box test would get
        wrong. Points exactly on an edge are not defined either way — inherent to the
        method, and not a distinction worth pinning down for a box centre.
        """
        points = self.points
        inside = False
        j = len(points) - 1
        for i, (xi, yi) in enumerate(points):
            xj, yj = points[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    def contains_box(self, box, width: int, height: int) -> bool:
        """Whether a pixel-space box's centre falls inside this zone.

        The centre, rather than any-corner or all-corners: a person is a tall box whose
        feet and head are often well outside a zone drawn around a work area, and
        all-corners would then never match while any-corner would match half the frame.
        The centre is the closest cheap stand-in for "where the subject is".
        """
        if not width or not height:
            return False
        try:
            x1, y1, x2, y2 = box
        except (TypeError, ValueError):
            return False
        return self.contains(((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height)


def zones_for_detector(payload_zones, detector: str) -> list:
    """The usable zones asking for ``detector``, from a ``detection_zones`` payload.

    Matched on the zone's `detectors` list, not its kind: one zone can ask for several
    things, and both kinds of zone can ask for any of them.

    An empty result means "no opinion, analyse the whole frame" — it is **not** "nothing
    matches, report nothing". Every caller has to honour that distinction: cameras that
    have never had zones drawn send no zones at all, and a silent detector on those would
    be far worse than an unfiltered one.
    """
    if not isinstance(payload_zones, list):
        return []

    zones = []
    for raw in payload_zones:
        zone = Zone.from_dict(raw)
        if zone is not None and detector in zone.detectors:
            zones.append(zone)
    return zones


def match(zones: list, box, width: int, height: int):
    """The first zone a box falls in, or None.

    ``None`` when ``zones`` is empty means the same as it does when nothing matched, so
    callers must check whether there were any zones *before* deciding a detection is out
    of scope — see :func:`zones_for_detector`.
    """
    for zone in zones:
        if zone.contains_box(box, width, height):
            return zone
    return None


def filter_by_zones(items, zones: list, box_of, width: int, height: int) -> tuple:
    """Split ``items`` into those inside a zone and those outside, with their zones.

    Returns ``(kept, dropped)`` where ``kept`` is a list of ``(item, zone)`` pairs. With no
    zones every item is kept against a zone of ``None``, which is how "no opinion" stays
    distinguishable from "matched a zone".

    An item whose box can't be read is **kept**, not dropped. A detector that stopped
    reporting because its boxes changed shape would be a silent failure, and silence is
    the worst outcome for a compliance or security finding.
    """
    if not zones:
        return [(item, None) for item in items], []

    kept, dropped = [], []
    for item in items:
        box = box_of(item)
        if box is None:
            kept.append((item, None))
            continue
        zone = match(zones, box, width, height)
        if zone is None:
            dropped.append(item)
        else:
            kept.append((item, zone))
    return kept, dropped


def should_notify(matched_zones: list, fallback: bool) -> bool:
    """Whether findings in ``matched_zones`` should notify.

    ``fallback`` is this app's own config switch, used when no zone was matched (because
    the camera sent none). A single zone asking for notification is enough: missing a real
    violation is a much worse failure than one extra message.
    """
    zones = [z for z in matched_zones if z is not None]
    if not zones:
        return fallback
    return any(z.notify for z in zones)
