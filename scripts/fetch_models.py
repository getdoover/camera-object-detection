#!/usr/bin/env python3
"""Fetch and prepare the ONNX weights this app ships.

Run on a dev machine, not the device -- exporting a YOLO .pt needs ultralytics and
torch, which never get installed into the device image. The resulting .onnx files in
models/ are committed so the image build (and a device with no internet) doesn't
depend on Hugging Face being reachable.

    uv run --group dev --with onnx scripts/fetch_models.py

Weights and licences
--------------------
* PPE (models/ppe.onnx) -- Hansung-Cho/yolov8-ppe-detection, MIT. 10 classes; we use
  person / hardhat / no-hardhat / safety vest / no-safety vest. Published as a .pt,
  so it gets exported to ONNX here.

  Picked over the two obvious alternatives by measuring all three, at conf=0.3, on
  four images from keremberke/construction-safety-object-detection plus one real
  frame from a site camera:

    model                          ds0  ds1  ds2  ds3   real frame
    Hansung-Cho (MIT)               4    3    1    6    person found
    leeyunjai/yolo11-ppe            4    0    0    3    nothing
    baskarmother (MIT)              1    1    1    1    nothing

  Only Hansung-Cho reliably emits `Person` boxes at all -- baskarmother returned a
  single box per image and never found a person, which makes per-person compliance
  impossible. Don't swap the weights without re-running that comparison: this app's
  whole output depends on people being detected, not just equipment.
* Plate detection (models/plate.onnx) -- morsetechlab/yolov11-license-plate-detection,
  AGPL-3.0. Ships ONNX directly, so it's a straight download.
* General objects (models/objects.onnx) -- Ultralytics YOLO11n trained on COCO (80
  classes), **AGPL-3.0**, same caveat as the plate detector below. Exported here with
  the same settings as the PPE model. Its class list is duplicated in
  common/detectors/objects.py (COCO_CLASSES) for the config schema; tests/test_objects.py
  checks the two agree, so re-run the tests after swapping it.
* Plate OCR (models/plate_ocr.onnx + .yaml) -- ankandrew's fast-plate-ocr
  `cct-xs-v1-global-model`, MIT. Copied out of its hub cache and committed, because
  the library resolves that cache from `Path.home()` with no env override: Lambda sets
  HOME=/tmp, so weights baked into /root/.cache are invisible there and OCR silently
  degrades to detect-but-never-read. Loading by explicit path fixes that.

NOTE the plate detector and the general objects model are AGPL-3.0 while this repo is
Apache-2.0. The device image distributes them onto customer Doovits, so this repo is kept
public; see the licence note in the README before closing it or swapping weights.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parents[1] / "models"

PPE_REPO = "Hansung-Cho/yolov8-ppe-detection"
PPE_WEIGHTS = "best.pt"
PPE_OUTPUT = "ppe.onnx"

PLATE_URL = (
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/"
    "resolve/main/license-plate-finetune-v1n.onnx"
)
PLATE_OUTPUT = "plate.onnx"

# Resolved by ultralytics from its own release assets.
OBJECTS_WEIGHTS = "yolo11n.pt"
OBJECTS_OUTPUT = "objects.onnx"

# Must match ObjectDetectionConfig.inference_size's default. The export bakes a
# fixed input size into the graph, so a mismatch at runtime fails the session.
IMAGE_SIZE = 640


def download(url: str, dest: Path):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f}MB)")


def fetch_plate_model():
    dest = MODELS_DIR / PLATE_OUTPUT
    if dest.exists():
        print(f"{PLATE_OUTPUT} already present, skipping.")
        return
    print(f"Fetching plate detector -> {PLATE_OUTPUT}")
    download(PLATE_URL, dest)


def fetch_ppe_model():
    dest = MODELS_DIR / PPE_OUTPUT
    if dest.exists():
        print(f"{PPE_OUTPUT} already present, skipping.")
        return

    print(f"Fetching PPE model -> {PPE_OUTPUT}")
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics + huggingface_hub are needed to export the PPE model.\n"
            "Run: uv run --group dev scripts/fetch_models.py"
        )

    checkpoint = hf_hub_download(repo_id=PPE_REPO, filename=PPE_WEIGHTS)
    print(f"  got checkpoint {checkpoint}")

    model = YOLO(checkpoint)
    print(f"  classes: {model.names}")
    # opset 12 is what onnxruntime 1.19+ handles without warnings on aarch64.
    # simplify=False keeps onnxsim (another heavy dep) out of the picture.
    #
    # dynamic=True matters: without it ultralytics pins H/W in the graph and
    # onnxruntime *rejects* any other input size ("Got invalid dimensions for input")
    # rather than resizing. The cloud processor's whole advantage is being able to run a
    # larger inference size than a Doovit can afford, so a fixed 640 graph would silently
    # cap it at the device's limit. `imgsz` remains the export-time reference shape.
    exported = model.export(
        format="onnx", imgsz=IMAGE_SIZE, opset=12, simplify=False, dynamic=True
    )
    shutil.move(str(exported), dest)
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f}MB)")


def fetch_objects_model():
    dest = MODELS_DIR / OBJECTS_OUTPUT
    if dest.exists():
        print(f"{OBJECTS_OUTPUT} already present, skipping.")
        return

    print(f"Exporting general objects model {OBJECTS_WEIGHTS} -> {OBJECTS_OUTPUT}")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is needed to export the objects model.\n"
            "Run: uv run --group dev --with onnx scripts/fetch_models.py"
        )

    model = YOLO(OBJECTS_WEIGHTS)
    # Same export as the PPE model, for the same reasons (see fetch_ppe_model).
    exported = model.export(
        format="onnx", imgsz=IMAGE_SIZE, opset=12, simplify=False, dynamic=True
    )
    shutil.move(str(exported), dest)
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f}MB)")


OCR_HUB_MODEL = "cct-xs-v1-global-model"
OCR_OUTPUT = "plate_ocr.onnx"
OCR_CONFIG_OUTPUT = "plate_ocr.yaml"


def fetch_ocr_model():
    """Copy fast-plate-ocr's weights into models/ so they load by explicit path.

    Its hub caches into `Path.home()/.cache/fast-plate-ocr`, hardcoded with no env
    override. That breaks in Lambda, which sets HOME=/tmp: an image with the cache baked
    into /root/.cache is ignored, the library re-downloads on every cold start, and if it
    can't reach the network OCR degrades to detection-only -- plates boxed, none read.
    Vendoring them removes HOME, the network and the cache from the runtime path.
    """
    dest = MODELS_DIR / OCR_OUTPUT
    config_dest = MODELS_DIR / OCR_CONFIG_OUTPUT
    if dest.exists() and config_dest.exists():
        print(f"{OCR_OUTPUT} already present, skipping.")
        return

    print(f"Fetching plate OCR -> {OCR_OUTPUT}")
    try:
        # Downloads into the hub cache as a side effect; we then copy it out.
        from fast_plate_ocr import LicensePlateRecognizer
        from fast_plate_ocr.inference import hub
    except ImportError:
        sys.exit("fast-plate-ocr is needed: uv run --group dev scripts/fetch_models.py")

    LicensePlateRecognizer(OCR_HUB_MODEL)
    cached = hub.MODEL_CACHE_DIR / OCR_HUB_MODEL
    onnx = next(cached.glob("*.onnx"), None)
    config = next(cached.glob("*.yaml"), None)
    if not onnx or not config:
        sys.exit(
            f"expected an .onnx and a .yaml under {cached}, found: "
            f"{[p.name for p in cached.iterdir()]}"
        )

    shutil.copy2(onnx, dest)
    shutil.copy2(config, config_dest)
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f}MB) and {config_dest.name}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fetch_ppe_model()
    fetch_plate_model()
    fetch_objects_model()
    fetch_ocr_model()
    print("\nDone. Commit the .onnx files in models/ so the image build is offline.")


if __name__ == "__main__":
    main()
