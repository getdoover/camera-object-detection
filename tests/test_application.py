"""Tests for the message-ingest side of the app.

The important behaviours here aren't about models at all: not analysing our own
output (which would loop forever), not analysing thumbnails or videos, and
respecting the reason filter.
"""

from types import SimpleNamespace

from object_detection.application import (
    ANALYSED_BY_KEY,
    ObjectDetectionApplication,
)


def attachment(filename, content_type="image/jpeg"):
    return SimpleNamespace(filename=filename, content_type=content_type, size=1, url="")


class TestImageAttachments:
    pick = staticmethod(ObjectDetectionApplication._image_attachments)

    def test_media_list_selects_full_size_only(self):
        """The thumbnail is the same scene at 640x360 -- analysing it too would
        double the CPU cost for a worse answer."""
        payload = {
            "media": [
                {
                    "name": "Preset1",
                    "file": "Preset1.jpg",
                    "thumbnail": "Preset1-thumbnail.jpg",
                }
            ]
        }
        attachments = [attachment("Preset1.jpg"), attachment("Preset1-thumbnail.jpg")]
        got = self.pick(payload, attachments)
        assert [name for name, _ in got] == ["Preset1"]
        assert [a.filename for _, a in got] == ["Preset1.jpg"]

    def test_multiple_views(self):
        payload = {
            "media": [
                {"name": "visible", "file": "visible.jpg"},
                {"name": "thermal", "file": "thermal.jpg"},
            ]
        }
        attachments = [attachment("visible.jpg"), attachment("thermal.jpg")]
        assert len(self.pick(payload, attachments)) == 2

    def test_skips_video(self):
        """Video-mode snapshots land on the same channel and can't be image-decoded."""
        payload = {"media": [{"name": "event", "file": "event.mp4"}]}
        assert self.pick(payload, [attachment("event.mp4", "video/mp4")]) == []

    def test_media_naming_a_missing_attachment(self):
        payload = {"media": [{"name": "a", "file": "gone.jpg"}]}
        assert self.pick(payload, [attachment("other.jpg")]) == []

    def test_falls_back_to_all_images_without_media(self):
        attachments = [attachment("a.jpg"), attachment("b.png"), attachment("c.mp4")]
        got = self.pick({}, attachments)
        assert [a.filename for _, a in got] == ["a.jpg", "b.png"]

    def test_no_attachments(self):
        assert self.pick({}, []) == []
        assert self.pick({}, None) == []

    def test_ignores_malformed_media_entries(self):
        payload = {"media": ["not-a-dict", {"no_file": 1}]}
        assert self.pick(payload, [attachment("a.jpg")]) == []


class TestIsImage:
    def test_suffixes(self):
        is_image = ObjectDetectionApplication._is_image
        assert is_image("a.jpg")
        assert is_image("a.JPEG")
        assert is_image("a.png")
        assert not is_image("a.mp4")
        assert not is_image("a")
        assert not is_image("")
        assert not is_image(None)


class TestAnnotatedFilename:
    def test_replaces_suffix_with_jpg(self):
        name = ObjectDetectionApplication._annotated_filename
        assert name("Preset1.jpg") == "Preset1-detected.jpg"
        assert name("frame.png") == "frame-detected.jpg"

    def test_extensionless(self):
        assert ObjectDetectionApplication._annotated_filename("frame") == (
            "frame-detected.jpg"
        )

    def test_does_not_collide_with_the_source(self):
        """It's published to the same channel, so it must not overwrite the original."""
        assert ObjectDetectionApplication._annotated_filename("a.jpg") != "a.jpg"


class TestLoopGuard:
    def test_our_own_marker_is_what_breaks_the_cycle(self):
        """This app publishes into the channel it subscribes to.

        Without the marker, each result would come back as a new snapshot and cost
        another model run, forever.
        """
        assert ANALYSED_BY_KEY not in {"reason": "schedule", "media": []}
        assert ANALYSED_BY_KEY in {ANALYSED_BY_KEY: "object_detection_1"}
