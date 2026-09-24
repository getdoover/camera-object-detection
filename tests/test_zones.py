"""Zones as a spatial filter over model findings.

The contract under test is shared with the camera app, which draws the zones and sends
them in the snapshot payload. The two apps deploy separately and share no package, so the
field names are a wire contract — `test_wire_contract_matches_camera_app` guards it from
this side.
"""

import types

from common import zones as zones_mod


def _sq(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _finding(box):
    """Something shaped like a Person/Plate: an object with .detection.box in pixels."""
    return types.SimpleNamespace(detection=types.SimpleNamespace(box=box))


# --- parsing ---


def test_zone_needs_three_points():
    """A malformed zone is dropped, not treated as a zone that matches nothing.

    A zone that matched nothing would silently swallow every detection in the frame, which
    looks exactly like the detector having died.
    """
    assert zones_mod.Zone.from_dict({"detectors": ["ppe"], "points": _sq(0, 0, 1, 1)}) is not None
    assert zones_mod.Zone.from_dict({"detectors": ["ppe"], "points": [[0, 0], [1, 1]]}) is None
    assert zones_mod.Zone.from_dict({"detectors": ["ppe"], "points": []}) is None
    assert zones_mod.Zone.from_dict({"kind": "ppe"}) is None
    assert zones_mod.Zone.from_dict("not a dict") is None
    # A point that isn't a pair drops the whole zone rather than half of it.
    assert zones_mod.Zone.from_dict({"detectors": ["ppe"], "points": [[0, 0], [1], [1, 1]]}) is None


def test_zones_are_selected_by_detector():
    """A zone wanting PPE says nothing about where plates matter, and vice versa."""
    payload = [
        {"kind": "intrusion", "detectors": ["ppe"],
         "points": _sq(0, 0, 0.5, 0.5), "name": "Work Area"},
        {"kind": "intrusion", "detectors": ["anpr"],
         "points": _sq(0.5, 0.5, 1, 1), "name": "Entry Lane"},
        # A zone asking for nothing is not ours to filter on, whatever its kind.
        {"kind": "intrusion", "detectors": [], "points": _sq(0, 0, 1, 1)},
        {"kind": "excluded_area", "points": _sq(0, 0, 1, 1)},
    ]
    ppe = zones_mod.zones_for_detector(payload, zones_mod.DETECTOR_PPE)
    anpr = zones_mod.zones_for_detector(payload, zones_mod.DETECTOR_ANPR)
    assert [z.name for z in ppe] == ["Work Area"]
    assert [z.name for z in anpr] == ["Entry Lane"]

    # No zones, or a payload that isn't a list, means "no opinion".
    assert zones_mod.zones_for_detector(None, zones_mod.DETECTOR_PPE) == []
    assert zones_mod.zones_for_detector([], zones_mod.DETECTOR_PPE) == []
    assert zones_mod.zones_for_detector("nonsense", zones_mod.DETECTOR_PPE) == []


def test_one_zone_can_ask_for_several_detectors():
    """One polygon can want a person's hard hat and any plate in the same frame."""
    payload = [
        {"kind": "excluded_area", "detectors": ["ppe", "anpr"],
         "points": _sq(0, 0, 1, 1), "name": "Yard"},
    ]
    assert [z.name for z in zones_mod.zones_for_detector(payload, "ppe")] == ["Yard"]
    assert [z.name for z in zones_mod.zones_for_detector(payload, "anpr")] == ["Yard"]


def test_an_older_camera_apps_ppe_kind_is_read_as_a_detector():
    """PPE/plates used to be zone kinds. Such a zone must keep being analysed.

    Ignoring it would silently drop the zone out of scope — the model would simply stop
    running over somebody's work area with nothing to show why.
    """
    payload = [{"kind": "ppe", "points": _sq(0, 0, 0.5, 0.5), "name": "Old Work Area"}]
    assert [z.name for z in zones_mod.zones_for_detector(payload, "ppe")] == [
        "Old Work Area"
    ]
    assert zones_mod.zones_for_detector(payload, "anpr") == []


def test_notify_defaults_to_true_when_unstated():
    """A zone that didn't state a preference must not be what silences an alert."""
    zone = zones_mod.Zone.from_dict({"detectors": ["ppe"], "points": _sq(0, 0, 1, 1)})
    assert zone.notify is True
    off = zones_mod.Zone.from_dict(
        {"kind": "ppe", "points": _sq(0, 0, 1, 1), "notify": False}
    )
    assert off.notify is False


# --- geometry ---


def test_box_centre_decides_membership():
    """A tall person box straddling the boundary belongs where its centre is.

    All-corners would never match a person whose feet fall outside a work-area zone;
    any-corner would match most of the frame.
    """
    zone = zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
    # 1000x1000 frame, so pixels map 1:1 onto tenths.
    assert zone.contains_box((100, 100, 300, 300), 1000, 1000) is True  # centre 0.2,0.2
    assert zone.contains_box((600, 600, 800, 800), 1000, 1000) is False  # centre 0.7,0.7
    # Straddling the boundary: the box extends well outside the zone but its centre is
    # inside, so it counts. This is the case that matters in practice - a standing person
    # in a work area whose feet cross the line.
    assert zone.contains_box((300, 300, 500, 600), 1000, 1000) is True  # centre 0.4,0.45

    # A centre landing exactly on an edge (here y=0.5) is deliberately undefined - ray
    # casting can't answer it and no sensible behaviour depends on the distinction. Asserted
    # only as "doesn't raise", so a future change of method isn't a spurious failure.
    assert zone.contains_box((300, 300, 500, 700), 1000, 1000) in (True, False)

    # A frame with no size can't be reasoned about; say no rather than divide by zero.
    assert zone.contains_box((0, 0, 10, 10), 0, 0) is False
    # A box that isn't four numbers is not a match, and doesn't raise.
    assert zone.contains_box(None, 1000, 1000) is False
    assert zone.contains_box((1, 2), 1000, 1000) is False


def test_concave_zone():
    """An L-shape, which a bounding-box test would get wrong in the notch."""
    zone = zones_mod.Zone(
        kind="ppe",
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.4, 0.4), (0.4, 1.0), (0.0, 1.0)],
    )
    assert zone.contains(0.1, 0.1) is True
    assert zone.contains(0.8, 0.2) is True
    assert zone.contains(0.2, 0.8) is True
    assert zone.contains(0.8, 0.8) is False


# --- filtering ---


def test_no_zones_keeps_everything():
    """The compatibility guarantee: a camera with no zones is analysed as it always was."""
    items = [_finding((10, 10, 20, 20)), _finding((900, 900, 950, 950))]
    kept, dropped = zones_mod.filter_by_zones(
        items, [], lambda i: i.detection.box, 1000, 1000
    )
    assert dropped == []
    assert [i for i, _z in kept] == items
    # Matched against no zone, which is how "no opinion" stays distinguishable from
    # "matched a zone that wants notifications".
    assert all(z is None for _i, z in kept)


def test_findings_outside_every_zone_are_dropped():
    zones = zones_mod.zones_for_detector(
        [{"detectors": ["ppe"], "points": _sq(0, 0, 0.5, 0.5), "name": "Work Area"}],
        zones_mod.DETECTOR_PPE,
    )
    inside, outside = _finding((100, 100, 200, 200)), _finding((800, 800, 900, 900))
    kept, dropped = zones_mod.filter_by_zones(
        [inside, outside], zones, lambda i: i.detection.box, 1000, 1000
    )
    assert [i for i, _z in kept] == [inside]
    assert dropped == [outside]
    assert kept[0][1].name == "Work Area"


def test_a_finding_with_no_box_is_kept():
    """Fail open. A detector whose boxes changed shape must not go silent.

    Silence is the worst outcome for a compliance finding: nobody notices a violation that
    was never reported, whereas an unfiltered one is merely noise.
    """
    zones = zones_mod.zones_for_detector(
        [{"detectors": ["ppe"], "points": _sq(0, 0, 0.5, 0.5)}], zones_mod.DETECTOR_PPE
    )
    boxless = types.SimpleNamespace(detection=types.SimpleNamespace(box=None))
    kept, dropped = zones_mod.filter_by_zones(
        [boxless], zones, lambda i: getattr(i.detection, "box", None), 1000, 1000
    )
    assert [i for i, _z in kept] == [boxless]
    assert dropped == []


# --- notification ---


def test_zone_overrides_the_global_switch_in_both_directions():
    loud = zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=_sq(0, 0, 1, 1), notify=True)
    quiet = zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=_sq(0, 0, 1, 1), notify=False)

    # A zone is the more specific statement, so it wins either way.
    assert zones_mod.should_notify([loud], fallback=False) is True
    assert zones_mod.should_notify([quiet], fallback=True) is False

    # No zone matched -> the app's own switch decides, exactly as before zones existed.
    assert zones_mod.should_notify([], fallback=True) is True
    assert zones_mod.should_notify([], fallback=False) is False
    assert zones_mod.should_notify([None], fallback=True) is True

    # One loud zone among several is enough - missing a real violation is the worse failure.
    assert zones_mod.should_notify([quiet, loud], fallback=False) is True


def test_zone_label():
    assert zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=[], name="Yard").label == "Yard"
    assert zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=[], id=3).label == "zone 3"
    assert zones_mod.Zone(kind="intrusion", detectors=["ppe"], points=[]).label == "zone"


def test_wire_contract_matches_camera_app():
    """The payload keys this app reads are the ones the camera app writes.

    Two separately deployed apps with no shared package, so nothing but a test stops one
    side renaming a field and silently disabling the filter. The camera app has the
    mirror of this check, in getdoover/cameras:
    `device_app/tests/test_imports.py::test_detector_zone_payload_is_what_object_detection_reads`.
    """
    # Exactly what camera_app.events.DetectionZone.to_dict() emits for a zone with a
    # detector on it.
    from_camera_app = {
        "id": 1,
        "enabled": True,
        "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "targets": ["person"],
        "kind": "intrusion",
        "detectors": ["ppe"],
        "notify": True,
        "name": "Work Area",
    }
    zone = zones_mod.Zone.from_dict(from_camera_app)
    assert zone is not None
    assert zone.detectors == ["ppe"]
    assert zone.kind == "intrusion"
    assert zone.notify is True
    assert zone.name == "Work Area"
    assert zone.id == 1
    assert len(zone.points) == 4
    # And the detectors the camera app can ask for are the ones we know about.
    assert set(zones_mod.KNOWN_DETECTORS) == {"ppe", "anpr"}


# --- friendly camera name ---


def test_notifications_use_the_cameras_display_name():
    """"Camera 2 detected ...", not "doover_camera_2 detected ...".

    The camera app publishes its display name with each snapshot. This is the consumer
    side of that key, and the pairing is a wire contract like `detection_zones` -- the two
    apps deploy separately and share no package.
    """
    from object_detection.application import ObjectDetectionApplication as App

    msg = types.SimpleNamespace(data={"camera_name": "Front Gate"})
    assert App._camera_name(msg, "doover_camera_2") == "Front Gate"

    # An older camera app sends no name -> the app key, exactly as before.
    assert App._camera_name(types.SimpleNamespace(data={}), "doover_camera_2") == (
        "doover_camera_2"
    )
    assert App._camera_name(types.SimpleNamespace(data=None), "doover_camera_2") == (
        "doover_camera_2"
    )
    # An empty name is not a name.
    assert App._camera_name(
        types.SimpleNamespace(data={"camera_name": ""}), "doover_camera_2"
    ) == "doover_camera_2"
