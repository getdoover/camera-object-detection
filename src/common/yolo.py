"""Minimal YOLOv8/v11 ONNX runner.

Deliberately hand-rolled rather than using ultralytics: ultralytics imports torch
unconditionally, and there is no torch build that fits comfortably on a Doovit
(Pi CM4, ~900MB RAM free alongside the other app containers). onnxruntime plus
this file is ~50MB and does the same job for inference-only use.

Everything here is synchronous and CPU-bound -- callers must push it to a thread
(``asyncio.to_thread``) so the app's event loop keeps servicing the DDA stream.
"""

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("OBJECT_DETECTION_MODEL_DIR", "models"))


@dataclass
class Detection:
    """One box, in *original image* pixel coordinates."""

    label: str
    confidence: float
    # x1, y1, x2, y2
    box: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": [x1, y1, x2, y2],
        }


class ModelUnavailable(Exception):
    """The weights file isn't present, so this detector can't run."""


def letterbox(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio and pad to a square ``size`` x ``size``.

    Returns the padded image, the scale factor applied, and the (left, top) pad so
    boxes can be mapped back to original coordinates.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, (left, top)


class YoloOnnx:
    """A YOLOv8/v11 detection model loaded from an ONNX file."""

    def __init__(self, path: Path, class_names: dict[int, str] | None = None):
        if not path.exists():
            raise ModelUnavailable(f"model weights not found at {path}")

        opts = ort.SessionOptions()
        # See the thread-pinning note in the Dockerfile: this model shares a 4-core
        # CPU with every other app on the device, so it gets one core.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.path = path
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.fixed_size = self._fixed_input_size(model_input.shape)
        self.class_names = class_names or self._names_from_metadata()
        log.info(
            f"Loaded {path.name} with {len(self.class_names)} classes: "
            f"{sorted(self.class_names.values())}"
            + (f", fixed {self.fixed_size}px input" if self.fixed_size else "")
        )

    @staticmethod
    def _fixed_input_size(shape) -> int | None:
        """The square input size baked into the graph, or None if it's dynamic.

        An ultralytics export pins H and W unless exported with ``dynamic=True``, and
        onnxruntime then *rejects* any other size outright rather than resizing —
        "Got invalid dimensions for input". Reading it here lets `detect` honour the
        model instead of trusting a config value the weights can't satisfy.
        """
        # NCHW; the H/W entries are ints when static and strings ('height') when not.
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return height
        return None

    def _names_from_metadata(self) -> dict[int, str]:
        """Read the class map ultralytics embeds in the ONNX metadata.

        It's stored as the *repr* of a python dict, not JSON, so it needs
        literal_eval rather than json.loads.
        """
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if not raw:
            log.warning(
                f"{self.path.name} has no 'names' metadata; detections will be "
                f"labelled by index."
            )
            return {}
        try:
            names = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            log.warning(f"Couldn't parse class names from {self.path.name}: {e}")
            return {}
        return {int(k): str(v).lower() for k, v in names.items()}

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.4,
        iou: float = 0.45,
        size: int = 640,
        wanted: set[str] | None = None,
    ) -> list[Detection]:
        """Run the model over a BGR image and return boxes in image coordinates.

        ``wanted`` filters by class name *after* NMS -- filtering before it would let
        a suppressed-by-a-better-overlapping-box detection survive.

        ``size`` is a request, not a guarantee: a model exported with a fixed input
        overrides it, because onnxruntime errors on a mismatch rather than adapting. Set
        it larger only for weights exported with dynamic axes (see
        ``scripts/fetch_models.py``).
        """
        if self.fixed_size and size != self.fixed_size:
            log.debug(
                f"{self.path.name} has a fixed {self.fixed_size}px input; ignoring the "
                f"requested {size}px."
            )
            size = self.fixed_size

        padded, scale, (pad_x, pad_y) = letterbox(image, size)
        # BGR->RGB, HWC->CHW, 0-1
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.input_name: blob})[0]
        boxes, scores, class_ids = self._decode(outputs, confidence)
        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou)
        if len(keep) == 0:
            return []

        h, w = image.shape[:2]
        detections = []
        for i in np.asarray(keep).flatten():
            bx, by, bw, bh = boxes[i]
            # Undo the letterbox: remove padding, then the resize.
            x1 = round((bx - pad_x) / scale)
            y1 = round((by - pad_y) / scale)
            x2 = round((bx + bw - pad_x) / scale)
            y2 = round((by + bh - pad_y) / scale)
            # A box can legitimately extend past the frame edge (a person half out
            # of shot); clamp rather than drop so the subject is still reported.
            box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

            cid = int(class_ids[i])
            label = self.class_names.get(cid, str(cid))
            if wanted and label not in wanted:
                continue
            detections.append(Detection(label, float(scores[i]), box))
        return detections

    @staticmethod
    def _decode(outputs: np.ndarray, confidence: float):
        """Pull boxes/scores/class ids out of the raw model output.

        YOLOv8/v11 emit ``(1, 4 + num_classes, num_anchors)``; some exporters
        transpose that to ``(1, num_anchors, 4 + num_classes)``. Anchors always
        vastly outnumber ``4 + num_classes``, so the longer axis is the anchor axis.
        """
        pred = np.squeeze(outputs, axis=0)
        if pred.shape[0] > pred.shape[1]:
            pred = pred.T
        # pred is now (4 + num_classes, num_anchors)

        class_scores = pred[4:, :]
        class_ids = np.argmax(class_scores, axis=0)
        scores = class_scores[class_ids, np.arange(class_scores.shape[1])]

        mask = scores >= confidence
        if not mask.any():
            return [], [], []

        # The model emits centre-form boxes, but cv2.dnn.NMSBoxes reads its input as
        # top-left (x, y, w, h). Feeding it centres shifts every box by half its own
        # size, which perturbs IoU differently for large and small boxes and quietly
        # mis-suppresses -- so convert to top-left here, and keep that form all the
        # way through to the reconstruction in detect().
        cx, cy, bw, bh = pred[0, mask], pred[1, mask], pred[2, mask], pred[3, mask]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
        return (
            boxes.tolist(),
            scores[mask].astype(float).tolist(),
            class_ids[mask],
        )
