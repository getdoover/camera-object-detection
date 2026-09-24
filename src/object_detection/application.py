"""Object detection over the camera apps' published snapshots.

One install per camera. The app is entirely event-driven: it subscribes to its camera
app's channel, and every time that camera publishes a snapshot message it fetches the
attached image, runs the enabled detectors, and **edits that message in place** with
the findings and an annotated copy — so a frame and its analysis are one timeline entry
rather than two a reader has to pair up. Each analysis is also recorded to tag history
(see ``object_detection_shared.tags``).

Each install loads its own copy of the models it has enabled, so on a Doovit with well
under a gigabyte to spare, enable only what each camera needs — see the README.

The inference itself lives in ``common``, shared verbatim with the cloud processor
variant (``object_detection_processor``) so the two can't drift apart.
"""

import asyncio
import logging
from datetime import datetime, timezone

from common import annotate as annotate_mod
from common import detectors as detectors_mod
from common import pipeline
from object_detection_shared.notifications import ObjectDetectionNotifications
from object_detection_shared.tags import ObjectDetectionTags, update_running_tags
from pydoover.docker import Application
from pydoover.models import (
    EventSubscription,
    File,
    MessageCreateEvent,
)

from .app_config import ObjectDetectionConfig

log = logging.getLogger()

# The camera app's `camera_event` channel -- the hook doover automations subscribe
# to. We publish onto it with our own kinds (`ppe_violation`, `anpr`,
# `object_detected`) so an automation can act on a finding exactly as it does for the
# camera's own events.
CAMERA_EVENT_CHANNEL = "camera_event"

# Marks a message as our own output. This app publishes into the very channel it
# subscribes to, so without a marker every result we publish would come straight
# back as a new snapshot to analyse -- an endless loop that also costs a model run
# each time round. Checked before anything else in the handler.
ANALYSED_BY_KEY = "analysed_by"

# Extensions we can decode. Video snapshots (the camera app's "Video" mode) land on
# the same channel and can't be run through an image decoder.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

ANNOTATED_SUFFIX = "-detected"
# Matches the camera app's convention (`<name>-thumbnail.jpg`), so its gallery treats our
# previews the same way as its own.
THUMBNAIL_SUFFIX = "-thumbnail"
# How the annotated frame is labelled in `media` -- its own view rather than replacing
# the source entry, so the unannotated frame stays browsable.
DETECTED_VIEW_SUFFIX = " (detected)"

# How long to wait after a snapshot message before re-reading it for its attachments.
# The device agent only knows the attachment URLs once its upload of the files has
# completed, so asking immediately gets nothing back. One second covers the local
# queue-and-upload on a healthy link; if it hasn't landed by then we log and skip the
# frame rather than block the event stream waiting for it.
ATTACHMENT_WAIT_SEC = 1


class ObjectDetectionApplication(Application):
    config: ObjectDetectionConfig
    tags: ObjectDetectionTags

    config_cls = ObjectDetectionConfig
    tags_cls = ObjectDetectionTags
    notifications_cls = ObjectDetectionNotifications

    async def setup(self):
        self.detectors = detectors_mod.load_enabled(self.config)
        log.info(f"Detectors running: {[d.name for d in self.detectors]}")

        if not self.detectors:
            log.warning(
                "No detectors are enabled (or none could load their weights) -- "
                "snapshots will be ignored."
            )

        # One model run at a time, so a PTZ camera's several views, or snapshots that
        # arrive back to back, queue rather than multiply peak RAM on a 4-core CM4
        # shared with the camera apps.
        self._inference_lock = asyncio.Lock()

        key = self.config.camera_app_key
        if not key:
            log.warning("No camera app configured; nothing to subscribe to.")
            return
        log.info(f"Subscribing to snapshots from '{key}'.")
        self.device_agent.add_event_callback(
            key, self.on_camera_message, EventSubscription.message_create
        )

    async def main_loop(self):
        # Everything happens in the subscription callbacks; the loop only exists to
        # surface that the app is alive and what it has done.
        log.info(
            f"Watching '{self.config.camera_app_key}'. "
            f"Analysed {self.tags.analysed_count.value} snapshot(s), "
            f"{self.tags.violation_count.value} PPE violation(s)."
        )

    # -- ingest ---------------------------------------------------------------

    async def on_camera_message(self, event: MessageCreateEvent):
        try:
            await self._handle_camera_message(event)
        except Exception as e:
            # A subscription callback that raises kills the stream for that channel,
            # taking every future snapshot with it. One bad frame must not do that.
            log.error(f"Failed to process camera message: {e}", exc_info=e)

    async def _handle_camera_message(self, event: MessageCreateEvent):
        message = event.message
        payload = message.data or {}
        app_key = event.channel.name

        if ANALYSED_BY_KEY in payload:
            # Our own result coming back round. See ANALYSED_BY_KEY.
            return

        reason = payload.get("reason")

        # The camera app sets this on its motion snapshots to say whether it wants them
        # analysed (its "Motion Snapshot Config > Object Detection" setting). It's
        # authoritative in both directions and overrides the reason filter: the camera
        # is the thing that knows whether this particular frame was captured to be
        # analysed, and honouring only the True case would leave no way to opt a single
        # camera out.
        wanted_by_camera = payload.get("object_detection")
        if wanted_by_camera is False:
            log.debug(f"'{app_key}' snapshot not marked for object detection.")
            return

        wanted = self.config.wanted_reasons
        if wanted_by_camera is not True and wanted and reason not in wanted:
            log.debug(f"Ignoring '{app_key}' snapshot (reason={reason}).")
            return

        if not self.detectors:
            return

        message = await self._await_attachments(app_key, message)
        targets = self._image_attachments(payload, message.attachments)
        if not targets:
            # Say so rather than returning quietly. A snapshot message whose payload
            # names media but carries no attachments used to look identical to "no
            # snapshots are arriving", which is how this went unnoticed.
            if isinstance(payload.get("media"), list) and payload["media"]:
                log.warning(
                    f"'{app_key}' published {len(payload['media'])} media item(s) but "
                    f"the message still carries no attachments, so there is nothing to "
                    f"analyse. Named: "
                    f"{[m.get('file') for m in payload['media'] if isinstance(m, dict)]}"
                )
            return

        # The camera app sends the zones that concern us along with the frame, so they
        # can't be out of step with it. Absent means "analyse the whole frame".
        zones = payload.get("detection_zones")

        log.info(
            f"Analysing {len(targets)} image(s) from '{app_key}' (reason={reason})."
        )
        for name, attachment in targets:
            await self._analyse_attachment(app_key, message, name, attachment, zones)

    async def _await_attachments(self, app_key: str, message):
        """Re-read the message so it carries its attachments.

        A ``MessageCreate`` event never has them: the publishing app hands its files to
        the device agent, which queues them for upload and mints no local URLs, so the
        event (and the agent's cached copy) lists none. The agent fills them in on
        ``GetMessage`` once the upload lands, so the sequence is: wait for the upload,
        then ask again.

        Returns the original message unchanged if the re-read fails or still has
        nothing — the caller logs that case, and a frame we can't reach is not worth
        raising over.
        """
        await asyncio.sleep(ATTACHMENT_WAIT_SEC)
        try:
            refetched = await self.device_agent.fetch_message(app_key, message.id)
        except Exception as e:
            log.warning(
                f"Couldn't re-read message {message.id} on '{app_key}' for its "
                f"attachments: {e}",
                exc_info=e,
            )
            return message

        if refetched is None or not refetched.attachments:
            return message
        return refetched

    @classmethod
    def _image_attachments(cls, payload: dict, attachments: list) -> list:
        """Pick the full-size images out of a snapshot message.

        The camera app publishes a ``media`` list naming which attachment is the
        full-size file and which is its thumbnail; the thumbnail is the same scene at
        640x360, so running the models over it as well would double the CPU cost to
        produce a worse answer. Where there's no ``media`` list (an older camera app,
        or another publisher) every image attachment is analysed.
        """
        by_filename = {a.filename: a for a in attachments or []}

        media = payload.get("media")
        if not isinstance(media, list):
            return [
                (a.filename, a) for a in attachments or [] if cls._is_image(a.filename)
            ]

        targets = []
        for entry in media:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("file")
            attachment = by_filename.get(filename)
            if attachment is None or not cls._is_image(filename):
                continue
            targets.append((entry.get("name") or filename, attachment))
        return targets

    @staticmethod
    def _is_image(filename: str) -> bool:
        return bool(filename) and filename.lower().endswith(IMAGE_SUFFIXES)

    async def _analyse_attachment(self, app_key, message, name, attachment, zones=None):
        try:
            file = await self.device_agent.fetch_message_attachment(attachment)
        except Exception as e:
            log.warning(
                f"Couldn't fetch '{attachment.filename}' from '{app_key}': {e}",
                exc_info=e,
            )
            return

        image = annotate_mod.decode(file.data)
        if image is None:
            log.warning(f"Couldn't decode '{attachment.filename}' as an image.")
            return

        async with self._inference_lock:
            analysis = await asyncio.to_thread(
                pipeline.analyse,
                self.detectors,
                image,
                self.config.inference_size.value,
                zones,
            )

        await self._publish_result(app_key, message, name, attachment, image, analysis)

    # -- publish --------------------------------------------------------------

    async def _publish_result(
        self, app_key, message, name, attachment, image, analysis
    ):
        # `findings` is what the models saw, unfiltered, so the annotated frame and the
        # timeline entry still show the whole picture. What the zones narrow is what gets
        # *reported* — the summary, the events and the notifications.
        report = pipeline.report(
            self.detectors, [analysis], self._camera_name(message, app_key)
        )

        await self.tags.analysed_count.set(self.tags.analysed_count.value + 1)
        await self._record_metrics(report.metrics)

        if not analysis.found_anything and not self.config.publish_clean_results.value:
            log.info(f"Nothing detected in '{attachment.filename}'.")
            return

        payload = {
            ANALYSED_BY_KEY: self.app_key,
            "analysed_at": datetime.now(tz=timezone.utc).isoformat(),
            "findings": {name: analysis.findings},
            "summary": report.summary,
        }

        files = []
        if self.config.annotate.value:
            try:
                annotated = analysis.annotate(image)
                filename = self._annotated_filename(attachment.filename)
                thumb_name = f"{filename.rsplit('.', 1)[0]}{THUMBNAIL_SUFFIX}.jpg"
                files.append(
                    File(
                        filename=filename,
                        content_type="image/jpeg",
                        size=0,
                        data=annotate_mod.encode_jpeg(annotated),
                    )
                )
                files.append(
                    File(
                        filename=thumb_name,
                        content_type="image/jpeg",
                        size=0,
                        data=annotate_mod.encode_thumbnail_jpeg(annotated),
                    )
                )
                # Put the annotated frame in `media` too, or a gallery driven off that
                # list never shows it -- the attachment would be there but invisible.
                # The whole list is rebuilt, camera entries included: a merge patch
                # replaces a list rather than appending, so sending only ours would drop
                # the original snapshot from the gallery.
                entry = {
                    "name": f"{name}{DETECTED_VIEW_SUFFIX}",
                    "file": filename,
                    "thumbnail": thumb_name,
                }
                existing = [
                    e
                    for e in (message.data or {}).get("media") or []
                    if isinstance(e, dict) and e.get("file") != filename
                ]
                payload["media"] = existing + [entry]
            except Exception as e:
                log.warning(f"Couldn't annotate the image: {e}", exc_info=e)

        # Edit the camera's own snapshot message rather than publishing a second one, so
        # a frame and its analysis are one timeline entry instead of two that a reader
        # has to pair up. Matches the cloud processor, which has to work this way: its
        # anti-recursion guard blocks create_message on the invoking channel.
        #
        # replace_data=False keeps the camera's payload (reason, media, night);
        # clear_attachments=False keeps the original snapshot beside the annotated copy.
        try:
            await self.device_agent.update_message(
                app_key,
                message.id,
                payload,
                files=files,
                replace_data=False,
                clear_attachments=False,
            )
        except Exception as e:
            log.error(
                f"Failed to update message {message.id} on '{app_key}': {e}", exc_info=e
            )

        await self._publish_events(app_key, report.events)
        for notification in report.notifications:
            # The declaration supplies the topic and severity; only the text varies.
            await self.notifications[notification.event].send(notification.text)

    @staticmethod
    def _annotated_filename(filename: str) -> str:
        stem, _, _suffix = filename.rpartition(".")
        if not stem:
            return f"{filename}{ANNOTATED_SUFFIX}.jpg"
        return f"{stem}{ANNOTATED_SUFFIX}.jpg"

    async def _record_metrics(self, metrics: dict):
        """Write this analysis's figures to tags, logged to history.

        Goes to the tag manager directly rather than through each bound tag, because
        a bound tag's set skips a value that hasn't changed — and "still 2 people"
        is a data point this history exists to hold. `last_analysed_at` goes with them
        so every point in the history is tied to an analysis.
        """
        values = {
            "last_analysed_at": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            **metrics,
        }
        await self.tag_manager.set_tags(
            {self.app_key: values}, only_if_changed=False, log=True
        )

    async def _publish_events(self, app_key, events):
        """Publish structured events for automations, mirroring `camera_event`."""
        now = datetime.now(tz=timezone.utc).isoformat()

        for _detector, event in events:
            kind = event["kind"]
            await self._publish_camera_event(
                kind,
                app_key,
                timestamp=now,
                **{k: v for k, v in event.items() if k != "kind"},
            )

        await update_running_tags(self.tags, events)

    async def _publish_camera_event(self, kind: str, app_key: str, **extra):
        payload = {
            "kind": kind,
            # The camera the finding came from, not this app -- an automation cares
            # which camera saw it.
            "app_key": app_key,
            "detected_by": self.app_key,
            **extra,
        }
        try:
            await self.create_message(CAMERA_EVENT_CHANNEL, payload)
        except Exception as e:
            log.warning(f"Failed to publish {kind} event: {e}", exc_info=e)

    @staticmethod
    def _camera_name(message, app_key: str) -> str:
        """What to call the camera in a message a person reads.

        The camera app publishes its display name with each snapshot; the app key is the
        fallback for an older camera app that doesn't. Only for human-facing text --
        `_publish_events` keeps using the app key, because an automation matches on the
        key and a display name can be renamed at any time.
        """
        return (
            ((message.data or {}).get("camera_name") or app_key) if message else app_key
        )
