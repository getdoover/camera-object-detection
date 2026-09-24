"""Draw detection results onto a frame.

Colour carries the meaning, so the timeline is readable at thumbnail size without
reading any text: red for a person missing required PPE, green for a compliant one,
amber for a plate, blue for an object a rule asked about.

Detectors don't pick colours — they say what each box *means* (``base.STYLE_*``) and
this module owns how that looks, so the meanings stay consistent across detectors.
"""

import cv2
import numpy as np

from .detectors.base import STYLE_BAD, STYLE_OBJECT, STYLE_OK, STYLE_PLATE

# BGR.
RED = (60, 60, 220)
GREEN = (80, 175, 80)
AMBER = (40, 170, 235)
BLUE = (200, 120, 40)
WHITE = (255, 255, 255)

COLOURS = {STYLE_BAD: RED, STYLE_OK: GREEN, STYLE_PLATE: AMBER, STYLE_OBJECT: BLUE}

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Box/label geometry is scaled off the image's own size so a 4K frame doesn't get
# hairline boxes and a 640px one doesn't get a label covering half the subject.
BASE_DIMENSION = 1080


def _scale(image) -> float:
    return max(0.4, min(2.5, max(image.shape[:2]) / BASE_DIMENSION))


def _draw_box(image, box, colour, label: str, scale: float):
    thickness = max(1, round(2 * scale))
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

    if not label:
        return

    font_scale = 0.6 * scale
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
    pad = round(4 * scale)

    # Prefer the label above the box, but drop it inside when the box is hard against
    # the top of the frame, otherwise it's drawn off-image and lost.
    top = y1 - text_h - baseline - pad * 2
    if top < 0:
        top = y1
    bottom = top + text_h + baseline + pad * 2

    cv2.rectangle(image, (x1, top), (x1 + text_w + pad * 2, bottom), colour, -1)
    cv2.putText(
        image,
        label,
        (x1 + pad, bottom - baseline - pad),
        FONT,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def annotate(image: np.ndarray, results) -> np.ndarray:
    """Return a copy of ``image`` with every result's annotations drawn on it."""
    out = image.copy()
    scale = _scale(out)
    for result in results:
        if result is None:
            continue
        for a in result.annotations():
            _draw_box(out, a.box, COLOURS.get(a.style, BLUE), a.label, scale)
    return out


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode the annotated image")
    return buf.tobytes()


def decode(data: bytes) -> np.ndarray | None:
    """Decode image bytes to BGR, or None if it isn't a decodable image."""
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


# The camera app's thumbnails are the camera's own 640x360 sub-stream picture, so match
# that width -- a timeline showing both side by side then gets consistent previews.
THUMBNAIL_WIDTH = 640


def encode_thumbnail_jpeg(image: np.ndarray, width: int = THUMBNAIL_WIDTH) -> bytes:
    """A downscaled JPEG of ``image``, preserving aspect ratio.

    Worth the few milliseconds: the whole point of drawing boxes is to see them, and a
    gallery renders the thumbnail. Without one, the annotated frame either doesn't
    preview at all or forces a full-size download to show a 200px tile. Unlike the
    camera app we can't take the camera's sub-stream picture here -- that would be an
    unannotated frame -- so it's a resize of what we drew.
    """
    height, source_width = image.shape[:2]
    if source_width <= width:
        return encode_jpeg(image)
    scaled_height = max(1, round(height * width / source_width))
    small = cv2.resize(image, (width, scaled_height), interpolation=cv2.INTER_AREA)
    return encode_jpeg(small)
