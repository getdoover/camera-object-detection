"""Object detection as a cloud processor (AWS Lambda).

The same models and the same compliance reasoning as the on-device app — both import
``common`` — but a different shape around them:

* **Invoked, not subscribing.** The platform delivers the camera's snapshot message; we
  don't watch a channel. So there's no camera-app list in config.
* **Edits the message in place** rather than publishing a second one, so a snapshot and
  its analysis are one timeline entry. ``ProcessorDataClient`` nudges this way too: its
  anti-recursion guard blocks ``create_message`` on the invoking channel but permits
  ``update_message``.
* **Attachments just work.** In the cloud they carry real URLs, so there's none of the
  device-side dance of waiting for an upload before the image is reachable.

What this variant does **not** buy is accuracy from a bigger inference size. That was
the assumption; measurement killed it. The weights are trained at 640, and on a real
site frame 960 lost a person that 640 found. Raising the size shifts object scale away
from the training distribution, so more CPU here buys throughput, not better detection.

The genuine wins are: no device RAM/CPU budget to share with the camera apps (so
heavier *weights* become possible when we have some), attachments that resolve without
waiting on an upload, and one timeline entry per snapshot.
"""

import logging
from datetime import datetime, timezone

from common import annotate as annotate_mod
from common import detectors as detectors_mod
from common import pipeline
from object_detection_shared.notifications import ObjectDetectionNotifications
from object_detection_shared.tags import ObjectDetectionTags, update_running_tags
from pydoover.models import File, MessageCreateEvent
from pydoover.processor import Application

from .app_config import ObjectDetectionProcessorConfig

log = logging.getLogger()

# Marks a message as already analysed. The invoking-channel guard means our update
# can't re-trigger us, so this is not loop protection like it is on-device -- it's
# idempotency, for a replay or a manual re-invoke.
ANALYSED_BY_KEY = "analysed_by"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ANNOTATED_SUFFIX = "-detected"

# Matches the camera app's convention (`<name>-thumbnail.jpg`), so its gallery treats
# our previews the same way as its own.
THUMBNAIL_SUFFIX = "-thumbnail"
# How the annotated frame is labelled in `media`. It goes in as its own view rather than
# replacing the source entry, so the unannotated frame stays browsable.
DETECTED_VIEW_SUFFIX = " (detected)"

# Loaded once per *container*, not per invocation.
#
# Lambda reuses a warm container across invocations but calls the handler (and so
# `setup`) each time, and building an onnxruntime session costs ~700ms per model. Held
# at module scope so only a cold start pays for it; the detectors are stateless between
# frames, so sharing them is safe.
_DETECTORS: dict = {}


class ObjectDetectionProcessor(Application):
    config: ObjectDetectionProcessorConfig
    config_cls = ObjectDetectionProcessorConfig
    tags: ObjectDetectionTags
    tags_cls = ObjectDetectionTags
    notifications_cls = ObjectDetectionNotifications

    def _detectors(self):
        """The enabled detectors, built once per warm container."""
        # Key on every detector setting, so a config change rebuilds rather than silently
        # reusing a detector built for the old values.
        key = tuple(
            (name, _config_key(getattr(self.config, name, None)))
            for name in detectors_mod.LOADERS
        )
        if _DETECTORS.get("key") != key:
            _DETECTORS.clear()
            _DETECTORS["key"] = key
            _DETECTORS["detectors"] = detectors_mod.load_enabled(self.config)
            log.info(
                f"Built detectors (cold start): "
                f"{[d.name for d in _DETECTORS['detectors']]}"
            )
        return _DETECTORS["detectors"]

    async def on_message_create(self, event: MessageCreateEvent):
        message = event.message
        payload = message.data or {}
        channel = event.channel.name

        if ANALYSED_BY_KEY in payload:
            return

        reason = payload.get("reason")
        wanted_by_camera = payload.get("object_detection")
        if wanted_by_camera is False:
            log.info(f"'{channel}' snapshot not marked for object detection.")
            return

        wanted = self.config.wanted_reasons
        if wanted_by_camera is not True and wanted and reason not in wanted:
            log.info(f"Ignoring '{channel}' snapshot (reason={reason}).")
            return

        detectors = self._detectors()
        if not detectors:
            log.warning("No detectors enabled or their weights failed to load.")
            return

        # Checked *after* the "nothing loaded at all" guard, so the two situations don't
        # produce the same log line -- "no detectors enabled" would be a lie when the
        # detectors are fine and this event simply doesn't call for them.
        #
        # The camera has already decided what it saw, so running a specialist model for
        # something else is work whose answer we don't trust anyway -- and worse than
        # wasted: on a *vehicle* event the PPE model returned a person at 0.49 that was
        # really a traffic cone, and produced a "missing hard hat" violation from it.
        if self.config.match_detectors_to_event.value:
            wanted = [
                d for d in detectors if detectors_mod.wanted_for_reason(d, reason)
            ]
            skipped = sorted(d.name for d in detectors if d not in wanted)
            if not wanted:
                log.info(
                    f"reason={reason} calls for none of the enabled detectors "
                    f"({sorted(d.name for d in detectors)}) — nothing to do."
                )
                return
            if skipped:
                log.info(f"reason={reason}: skipping {skipped}.")
            detectors = wanted

        targets = self._image_attachments(payload, message.attachments)
        if not targets:
            log.info(f"No analysable image on '{channel}' message {message.id}.")
            return

        # The camera app sends the zones that concern us with the frame itself. Absent
        # means "analyse the whole frame", which is every camera that has never had zones
        # drawn on it.
        zones = payload.get("detection_zones")

        # One frame per message in practice; a PTZ camera contributing several presets
        # is analysed in order and the findings merged under their view names.
        findings, files, media, analyses = {}, [], [], []
        for name, attachment in targets:
            result = await self._analyse(attachment, detectors, name, zones)
            if result is None:
                continue
            analysis, view_files, media_entry = result
            findings[name] = analysis.findings
            analyses.append(analysis)
            files.extend(view_files)
            if media_entry:
                media.append(media_entry)

        if not findings:
            return

        # The camera's display name where there is one, for text a person reads. The
        # channel name is the fallback for an older camera app that doesn't send it.
        report = pipeline.report(
            detectors, analyses, payload.get("camera_name") or channel
        )
        await self._record_tags(report)
        await self._publish(channel, message, payload, findings, files, media, report)

    async def _record_tags(self, report):
        """This analysis's figures, logged to history; then the running tags.

        Buffered and committed by pydoover at the end of the invocation. Each set asks
        for a log, so a repeated value ("still 2 people") is recorded too.
        """
        await self.tags.analysed_count.set(self.tags.analysed_count.value + 1)
        values = {
            "last_analysed_at": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            **report.metrics,
        }
        for key, value in values.items():
            await self.tag_manager.set_tag(key, value, log=True)
        await update_running_tags(self.tags, report.events)

    async def _analyse(self, attachment, detectors, name, zones=None):
        try:
            data = await self.api.fetch_message_attachment(attachment)
        except Exception as e:
            log.warning(f"Couldn't download '{attachment.filename}': {e}", exc_info=e)
            return None

        image = annotate_mod.decode(data)
        if image is None:
            log.warning(f"Couldn't decode '{attachment.filename}' as an image.")
            return None

        # Zones narrow what gets *reported*, not what the models saw: `findings` stays
        # unfiltered so the annotated frame and the timeline entry still show everything.
        analysis = pipeline.analyse(
            detectors, image, self.config.inference_size.value, zones
        )

        files, media_entry = [], None
        if self.config.annotate.value:
            try:
                drawn = analysis.annotate(image)
                filename = self._annotated_filename(attachment.filename)
                thumb_name = f"{filename.rsplit('.', 1)[0]}{THUMBNAIL_SUFFIX}.jpg"
                files.append(
                    File(
                        filename=filename,
                        content_type="image/jpeg",
                        size=0,
                        data=annotate_mod.encode_jpeg(drawn),
                    )
                )
                files.append(
                    File(
                        filename=thumb_name,
                        content_type="image/jpeg",
                        size=0,
                        data=annotate_mod.encode_thumbnail_jpeg(drawn),
                    )
                )
                # Same shape as the camera app's own media entries, so a gallery renders
                # this without special-casing us. Named as its own view rather than
                # replacing the source entry, so the original frame stays browsable.
                media_entry = {
                    "name": f"{name}{DETECTED_VIEW_SUFFIX}",
                    "file": filename,
                    "thumbnail": thumb_name,
                }
            except Exception as e:
                log.warning(f"Couldn't annotate the image: {e}", exc_info=e)

        return analysis, files, media_entry

    async def _publish(self, channel, message, payload, findings, files, media, report):
        """Merge the findings into the original message and attach the annotation.

        `replace_data=False` so the camera's own payload survives, and
        `clear_attachments=False` so the original snapshot stays alongside the
        annotated copy rather than being replaced by it.
        """
        detail = {
            ANALYSED_BY_KEY: self.app_key,
            "analysed_at": datetime.now(tz=timezone.utc).isoformat(),
            "findings": findings,
            "summary": report.summary,
        }
        if media:
            # Send the *whole* list, camera entries included. A merge patch replaces a
            # list wholesale rather than appending to it, so sending only our entries
            # would drop the original snapshot out of the gallery -- attached, but
            # invisible. Rebuilt here so the result is right either way.
            detail["media"] = self._merged_media(payload, media)
        try:
            await self.api.update_message(
                channel_name=channel,
                message_id=message.id,
                data=detail,
                replace_data=False,
                files=files or None,
                clear_attachments=False,
            )
        except Exception as e:
            log.error(f"Failed to update message {message.id}: {e}", exc_info=e)
            return

        log.info(f"Updated message {message.id} on '{channel}': {detail['summary']}")
        # Notifications come from the zone-filtered report, never from `findings`:
        # `findings` is deliberately unfiltered (it backs the annotated frame), so
        # deriving them from it would notify about violations the zones excluded.
        for notification in report.notifications:
            # The declaration supplies the topic and severity; only the text varies.
            await self.notifications[notification.event].send(notification.text)

    @staticmethod
    def _merged_media(payload: dict, new_entries: list) -> list:
        """The camera's media entries plus ours, without duplicating on a re-run.

        Keyed by filename so re-analysing a message replaces our previous entry rather
        than appending a second copy of it.
        """
        existing = [e for e in (payload.get("media") or []) if isinstance(e, dict)]
        ours = {e["file"] for e in new_entries}
        return [e for e in existing if e.get("file") not in ours] + new_entries

    @classmethod
    def _image_attachments(cls, payload: dict, attachments: list) -> list:
        """Pick the full-size images out of a snapshot message.

        Mirrors the on-device app: the camera's ``media`` list says which attachment is
        full-size and which is its 640x360 thumbnail, and analysing the thumbnail too
        would double the cost for a worse answer. Also skips anything we've already
        attached ourselves, so a re-invoke doesn't analyse its own annotation.
        """
        by_filename = {a.filename: a for a in attachments or []}

        media = payload.get("media")
        if not isinstance(media, list):
            return [
                (a.filename, a)
                for a in attachments or []
                if cls._is_image(a.filename) and ANNOTATED_SUFFIX not in a.filename
            ]

        targets = []
        for entry in media:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("file")
            attachment = by_filename.get(filename)
            if attachment is None or not cls._is_image(filename):
                continue
            if ANNOTATED_SUFFIX in filename:
                continue
            targets.append((entry.get("name") or filename, attachment))
        return targets

    @staticmethod
    def _is_image(filename: str) -> bool:
        return bool(filename) and filename.lower().endswith(IMAGE_SUFFIXES)

    @staticmethod
    def _annotated_filename(filename: str) -> str:
        stem, _, _suffix = filename.rpartition(".")
        return f"{stem or filename}{ANNOTATED_SUFFIX}.jpg"


def _config_key(element):
    """A hashable snapshot of a config section's values, for the detector cache."""
    if element is None:
        return None
    children = getattr(element, "_elements", None)
    if isinstance(children, dict):
        return tuple((name, _config_key(c)) for name, c in sorted(children.items()))
    if isinstance(children, list):
        return tuple(_config_key(c) for c in children)
    try:
        return element.value
    except ValueError:
        return None
