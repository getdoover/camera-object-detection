# Object Detection

<img src="https://doover.com/wp-content/uploads/Doover-Logo-Landscape-Navy-padded-small.png" alt="Doover Logo" style="max-width: 300px;">

**Hard-hat / high-vis compliance, number-plate recognition and rule-driven general object
detection for camera snapshots, on-device or in the cloud**

<br/>

## Two variants, one inference core

| | `object_detection` (DEV) | `object_detection_processor` (PRO) |
|---|---|---|
| Runs on | the Doovit | AWS Lambda |
| Triggered by | subscribing to its camera app's channel | invoked by the platform, via a subscription to that channel |
| Gets the image | waits ~1s, re-reads the message (see below) | attachment URL, works immediately |
| Result | edits the snapshot message in place | edits the snapshot message in place |
| Inference size | 640 | 640 — bigger is *not* better, see below |
| Uploads every frame? | **no** — analysis is local | **yes** — cloud inference needs the upload |

Both import **`src/common/`**, which holds all the actual inference — `yolo.py`, the
`detectors/`, `pipeline.py`, `annotate.py` — and has zero doover coupling: everything there
takes and returns numpy arrays. The two app shells own only config, triggering and
publishing, so the models and the compliance reasoning can't drift between device and
cloud.

Pick per site. The device variant when bandwidth or privacy means boring frames
shouldn't leave the site; the processor when uploading is fine and you want the
accuracy that a real CPU buys.

### It works with the camera app, in a separate repo

The camera app lives in [getdoover/cameras](https://github.com/getdoover/cameras). The two
share no code, only the snapshot message: this app reads `media`, `reason`,
`object_detection`, `detection_zones` and `camera_name` from it, and marks its own edits
with `analysed_by`. The zone editor's detector values (`ppe`, `anpr`, `objects`) are also
defined on the camera side. Both repos have tests pinning their half of that contract
(`tests/test_zones.py` here), so a change to any of those keys needs a matching change
in the other repo.

This app is kept in its own public repository because of its model weights' AGPL-3.0
licence — see [Models](#models).

<br/>

## Overview

**Install it once per camera.** Every time that camera publishes a snapshot, the app
fetches the image, runs the enabled detectors over it, and publishes the result **back
to the camera's own channel** — so the finding lands in the camera's timeline next to the
picture it came from, with an annotated copy attached — and records the figures to [tag
history](#tag-history).

One install per camera is what lets each camera have its own detectors, thresholds and
rules: the gate camera reads plates, the yard camera checks PPE and counts cattle.

The models are ONNX files baked into the image, executed by onnxruntime on the CPU;
there is no API key. On the device variant that CPU is the Doovit's, so see
[Performance](#performance) for what several installs on one device cost.

<br/>

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| **Camera App** (device) / **Camera Channel** (cloud) | The camera to analyse — its app key | `doover_camera_1` |
| **PPE › Enabled** | Run hard-hat / high-vis detection | `false` |
| **PPE › Require Hard Hat** | Flag a person not wearing a hard hat | `true` |
| **PPE › Require High-Vis** | Flag a person not wearing a high-vis vest | `true` |
| **PPE › Minimum Confidence** | Drop detections below this confidence (0–100) | `55` |
| **PPE › Notify On Violation** | Notify when someone is missing required PPE | `true` |
| **ANPR › Enabled** | Detect and read vehicle number plates | `false` |
| **ANPR › Minimum Confidence** | Drop plate detections below this confidence | `40` |
| **ANPR › Minimum Plate Characters** | Discard OCR reads shorter than this | `4` |
| **ANPR › Notify On Plate Read** | Notify on every plate read | `false` |
| **Object Detection › Enabled** | Run the general-purpose COCO model | `false` |
| **Object Detection › Rules** | What to look for, each with its own **Minimum Confidence** (default `50`) — see [General object detection](#general-object-detection) | *(none)* |
| **Analyse Snapshots Because Of** | Only analyse snapshots with these `reason`s. Empty = everything | *(all)* |

| **Annotate Images** | Draw labelled boxes and publish the annotated frame | `true` |
| **Publish Results With No Findings** | Publish even when nothing was detected | `false` |
| **Inference Size** | Square size (px) frames are letterboxed to | `640` |
| **Match Detectors To Event** (cloud only) | Skip PPE on vehicle events and plates on person events | `true` |

Every **Minimum Confidence** is shown by default rather than tucked under advanced
settings: it's the setting most worth tuning per camera. Read [why PPE defaults to
55](#false-positives-and-why-the-default-confidence-is-55) before lowering it.

### The camera app can opt frames in or out

The camera app's **Motion Snapshot Config › Object Detection** setting puts
`"object_detection": true|false` on its motion snapshots. That flag is **authoritative
in both directions and overrides the reason filter above** — the camera is the thing
that knows whether a given frame was captured to be analysed, and honouring only the
`true` case would leave no way to opt one camera out. Snapshots with no flag fall
through to the reason filter as normal.

### Detection zones narrow *where* a finding counts

Zones are drawn in the camera app's zone editor, not here. The camera app sends the ones
that concern us — those whose `detectors` list names `ppe`, `anpr` or `objects` — on each
snapshot message as `detection_zones`, so they arrive with the frame they apply to and
can't be out of step with it. (The other two kinds, `intrusion` and `excluded_area`, are camera rules the
camera evaluates itself and never reach this app.)

```json
"detection_zones": [
  {"id": 1, "kind": "ppe",  "points": [[0.0,0.0],[0.5,0.0],[0.5,1.0],[0.0,1.0]],
   "notify": true,  "name": "Work Area"},
  {"id": 1, "kind": "anpr", "points": [[0.5,0.0],[1.0,0.0],[1.0,1.0],[0.5,1.0]],
   "notify": false, "name": "Entry Lane"}
]
```

Each detector is filtered by its own zones — a PPE zone says nothing about where plates
matter — and a zone's `notify` only speaks for the findings that fell inside it, so a loud
plate zone can't make a quiet PPE setting notify. A finding counts when its **box centre** falls inside a zone: a person is a tall
box whose feet and head are often outside a zone drawn round a work area, so requiring
every corner would never match, and any corner would match half the frame.

The models still run over the **whole frame**, and `findings` stays unfiltered so the
annotated image and the timeline entry show everything. Zones only narrow what gets
*reported* — the summary, the `camera_event`s and the notifications.

A matching zone's `notify` **overrides** the config switches above, in both directions: a
zone with it on speaks up even when **Notify On Violation** is off, and a zone with it off
stays quiet even when it's on. A zone is the more specific statement.

Two deliberate fail-open rules:

- **No zones of that kind** (including every camera that's never had zones drawn) means
  *analyse the whole frame and use the config switch* — **not** "nothing qualifies".
- **A finding with no readable box is kept**, not dropped.

Both exist because a silent detector is far worse than an unfiltered one: nobody notices
a violation that was never reported. A zone that filters everything out is logged for the
same reason — it otherwise looks identical to a detector that has stopped working.

<br/>


## What it publishes

### To the camera's own channel (the timeline)

A message shaped like the camera app's own snapshot messages, so an existing gallery
renders it without special-casing:

```json
{
  "reason": "object_detection",
  "analysed_by": "object_detection_1",
  "source": {"message_id": "…", "file": "Preset1.jpg", "name": "Preset1", "reason": "intruder"},
  "summary": "1 person missing hard hat, high vis",
  "media": [{"name": "Preset1", "file": "Preset1-detected.jpg"}],
  "findings": {
    "ppe": {
      "people": [{"box": [1509, 814, 1920, 1080], "confidence": 0.5,
                  "hard_hat": false, "high_vis": null, "implied": false}],
      "violations": [{"box": [1509, 814, 1920, 1080], "missing": ["hard_hat", "high_vis"]}]
    },
    "anpr": {"plates": [{"box": [], "confidence": 0.9, "plate": "ABC123"}]},
    "objects": {"objects": [{"label": "cow", "confidence": 0.82, "box": [40, 300, 410, 620]}]}
  }
}
```

`analysed_by` is not decoration — this app publishes into the very channel it
subscribes to, and that key is what stops each result coming back as a new snapshot
to analyse. Anything carrying it is skipped on ingest.

### To `camera_event` (automations)

The same channel the camera app uses, so automations hook off findings the same way:

```json
{"kind": "ppe_violation", "app_key": "doover_camera_1", "detected_by": "object_detection_1",
 "timestamp": "…", "count": 1, "missing": ["hard_hat"]}
```

`kind` is `ppe_violation`, `anpr` (which also carries `plate` and `confidence`), or
`object_detected` — one per triggered rule:

```json
{"kind": "object_detected", "app_key": "doover_camera_1", "detected_by": "object_detection_1",
 "timestamp": "…", "rule": "Cattle in laneway", "count": 3, "objects": ["cow"]}
```

### To tag history

Every analysis writes these tags **and logs them to history** — zeros and repeats
included, so the history has a point per analysed frame rather than one per change. A
detector that isn't enabled leaves its tags alone, and one that failed on a frame
records nothing for it (zeros would claim it looked).

| Tag | What it holds |
|---|---|
| `last_analysed_at` | Epoch ms of the analysis the other figures describe |
| `ppe_people` | People seen in the frame (whole frame) |
| `ppe_violations` | People missing required PPE, after zones |
| `anpr_plates` / `anpr_plates_read` | Plates seen (whole frame) / plates read, after zones |
| `objects_count` | Objects seen that some rule asks about (whole frame) |
| `rule_<name>` | One per object rule: how many of its objects were in its zones — recorded even below the rule's minimum count |
| `<detector>_max_confidence` / `<detector>_mean_confidence` | Over what that detector saw, 0–1; `0` when it saw nothing |

The "seen" counts are the whole frame, matching `findings`; the "after zones" ones match
what was reported. A rule's tag name is its name slugged (`Cattle in laneway` →
`rule_cattle_in_laneway`), so **renaming a rule starts a new series**.

Running tags are kept too, for dashboards: `analysed_count`, `violation_count`,
`last_ppe_violation` (epoch ms, same as the camera app's tag) and `last_plate`.

<br/>

## General object detection

Alongside the two specialists, one general-purpose model — YOLO11n trained on COCO, 80
everyday classes (people, vehicles, animals, bags, …) — runs once per frame, and any number
of **rules** read its output:

| Rule | Objects | Minimum Count | Notify |
|---|---|---|---|
| Cattle in laneway | `cow` | 1 | yes |
| Crowding at the gate | `person` | 6 | yes |
| Dog on site | `dog` | 1 | no |

A rule triggers when at least *Minimum Count* of its objects are in frame above its
confidence. Each triggered rule adds to the summary, publishes an `object_detected` event
and, if its **Notify** is on (or a zone says so), sends a notification.

**Rules are free; models are not.** The cost is per *model run* (~1.4s a frame for the PPE
model on a CM4), not per class, so every rule reads the same single pass. That's the reason
for one general model plus rules, rather than a model per object type. Only detections a
rule asked about are drawn and published — a COCO model finds chairs, cups and plants in
most frames, and showing them would bury what the rule is for.

**Counting happens after the zones.** A rule wanting two cows in the laneway won't fire on
one cow in the laneway and one in the paddock. Zones name `objects` in their `detectors`
list to scope the rules.

**What it's bad at is *absence*.** A general model has no idea of "a person *without* a hard
hat", which is why PPE and plates stay as their own detectors with their own reasoning. On
the cloud processor, **Match Detectors To Event** gates only those specialists: the camera
classifies people and vehicles, which says nothing about whether a cow is in frame, so
object rules run on every event.

### Adding a detector

Each detector is a module in `common/detectors/` implementing `base.Detector`: `analyse` a
frame into a result (what was seen, what's reportable, what to draw), then turn the
zone-filtered reportable items into a summary, events and alerts. Register its loader in
`detectors.LOADERS` and give both app configs a section of the same name. Nothing in the
app shells or `common/pipeline.py` names a detector.

<br/>

## How compliance is decided

The model detects **equipment, not compliance**. It emits independent boxes for
`person`, `hardhat`/`no-hardhat` and `safety vest`/`no-safety vest`, and the app turns
that into a per-person verdict. Two things there are easy to get wrong:

**Equipment is matched to people by containment, not IoU.** A hard hat is a tiny box
inside a large person box, so their IoU is near zero even when it's obviously being
worn. What matters is how much of the *equipment* box falls inside the person box.
There's a test pinning this (`test_iou_would_fail_here`) precisely because IoU looks
like the obvious choice.

**A positive detection always beats a negative one.** These models routinely emit both
`hardhat` and `no-hardhat` over the same head at similar confidence. So a person is
compliant if a *positive* box matches them, and is only flagged when none does. The
`no-*` classes are corroboration and a way to catch people the `person` class missed —
never the sole basis for a violation.

When the model sees a bare head or a vestless torso but no `person`, that equipment box
stands in for a person (`"implied": true`) so the violation isn't silently dropped.

> **Camera placement matters more than any setting here.** These models are trained on
> upright, roughly eye-level site footage. On a wide-angle camera mounted rotated, or
> looking steeply down, they miss people that a general-purpose COCO model finds
> easily. If detection seems poor, check the framing before touching confidence.

### False positives, and why the default confidence is 55

Measured on a live 4K frame from an AcuSense overlooking a yard, with no person in shot:

```
raw: [('safety vest', 0.69), ('person', 0.49)]   -> 1 violation: "NO HARD HAT"
```

Both boxes were an **orange traffic cone sitting on a wall**. Orange reads as hi-vis,
and the cone-plus-bin shape read as a low-confidence person. Sweeping the threshold on
that same frame:

| Confidence | Detections | Violations |
|---|---|---|
| 35, 40 | `safety vest` 0.69, `person` 0.49 | **1 (false)** |
| 50, 55, 60 | `safety vest` 0.69 | 0 |
| 70 | none | 0 |

The phantom person dies at 50, so the default is **55**. The stray `safety vest` survives
to ~70 and doesn't matter — a positive-PPE box with no person to attach to is discarded,
because only the `no-*` classes can imply a person.

**But raising the threshold is not a free win, and no threshold fixes this properly.**
Measured at 55 across the same set:

| Frame | Expected | At 55 |
|---|---|---|
| worker in hat + vest (`person` 0.84) | 1 person, 0 violations | ✅ as expected |
| worker in hat + vest (`person` 0.83) | 1 person, 0 violations | ✅ as expected |
| traffic cone on a wall (`person` 0.49) | nothing | ✅ nothing |
| **real person, no PPE (`person` 0.50)** | 1 violation | ❌ **missed** |

A genuine person scored **0.50** and a traffic cone **0.49**. They are not separable by
confidence — anything that suppresses the cone also suppresses that person. Both of those
frames come from wide-angle cameras mounted rotated and looking steeply down; the two
correctly-handled frames are ordinary upright site photos, where the model is confident
(0.83–0.84) and 55 has plenty of margin.

So the default trades a missed detection in bad framing for not crying wolf, and **the
real fix is the camera, not the number**. Point it so people appear upright and
reasonably large and the model moves to the 0.8s, well clear of the noise. If you must
run awkward framing, drop confidence to ~40 and expect orange plant, cones, bins and
signage to trigger it — high-vis detection is partly colour-driven.

<br/>

## Inference size is not an accuracy dial

Measured on two real frames, PPE model, conf 40:

| Frame | @640 | @960 | @1280 |
|---|---|---|---|
| upright worker in hat + vest | person **0.84** | person 0.85 | person 0.62 |
| hard wide-angle frame, real person | person **0.50** | **nothing found** | person 0.58 |

**960 lost a person that 640 found.** These weights are trained at 640, so moving the
input size shifts object scale away from the training distribution — it is not a
monotonic improvement, and 1280 costs ~4× the CPU for no reliable gain.

So both variants default to **640**, and the cloud variant's advantage is *not* a bigger
input. If PPE accuracy needs to improve, the lever is **heavier weights** (an `m`/`l`
fine-tune rather than this `n`-class one), which is what having a real CPU actually
unlocks — not a bigger number here.

The models are exported with `dynamic=True` so a larger size is *possible* to
experiment with; without that, ultralytics pins H/W in the graph and onnxruntime rejects
any other size outright rather than resizing. `YoloOnnx` reads the graph's input shape
and overrides a mismatched request, so a fixed-shape model can't be misconfigured into
a crash.

<br/>

## Number plates need a camera pointed at a choke point

Measured on a real yard camera (1280×720, wide-angle, nearest vehicle ~15m away):

| Inference size | conf 40 | conf 25 | Time/frame |
|---|---|---|---|
| 640 | 0 plates | 0 plates | 1.6 s |
| 960 | 0 plates | 0 plates | 3.0 s |
| 1280 | 0 plates | 1 plate, **23×17 px**, unreadable | 5.9 s |

Three utes were in shot, all with plates, and the best the detector managed was a
23-pixel-wide box that OCR could not read — `MIN_CROP_WIDTH` correctly rejected it
rather than inventing a plate. Raising **Inference Size** does not fix this: it costs
~4× the CPU from 640 to 1280 and still yields nothing legible.

**This is optics, not configuration.** A plate wants roughly 100+ px of width to read,
so ANPR needs a camera aimed at a gate or driveway choke point where vehicles pass close
and roughly square-on. A wide-angle camera surveying a whole yard will classify the
*vehicle* happily and never read its plate. Point one camera at the entrance for plates
and leave the yard camera to PPE.

<br/>

## Models

| Purpose | Weights | Licence |
|---|---|---|
| PPE (`models/ppe.onnx`) | [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) | MIT |
| Plate detection (`models/plate.onnx`) | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (v1n) | **AGPL-3.0** |
| General objects (`models/objects.onnx`) | [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/), COCO | **AGPL-3.0** |
| Plate OCR | [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) `cct-xs-v1-global-model`, cached at image build | MIT |

The `.onnx` files are committed so the image build — and a device with no internet —
doesn't depend on Hugging Face. Regenerate them with:

```bash
uv run --group dev --with onnx scripts/fetch_models.py
```

The PPE model was chosen by measuring three candidates at `conf=0.3` on four images
from `keremberke/construction-safety-object-detection` plus a real site frame:

| Model | ds0 | ds1 | ds2 | ds3 | Real frame |
|---|---|---|---|---|---|
| **Hansung-Cho (MIT)** | 4 | 3 | 1 | 6 | person found |
| leeyunjai/yolo11-ppe | 4 | 0 | 0 | 3 | nothing |
| baskarmother (MIT) | 1 | 1 | 1 | 1 | nothing |

Only Hansung-Cho reliably emits `Person` boxes at all; baskarmother returned one box
per image and never found a person, which makes per-person compliance impossible.
**Don't swap the weights without re-running that comparison** — the app's entire
output depends on people being detected, not just equipment.

### Safety glasses and steel-cap boots: why they aren't here

Asked for, and not currently possible. Recorded so the search isn't repeated.

**Glasses** — a model with the class exists: `Hexmon/vyra-yolo-ppe-detection`, YOLOv8m,
CC-BY-4.0, 14 classes including `Goggles`/`NO-Goggles`. Measured on the same frames as
the table above, at conf 0.35, it is much worse where it counts:

| Frame | current weights | vyra 14-class |
|---|---|---|
| worker in hat + vest | vest 0.89, hardhat 0.88, **person 0.84** | hardhat 0.47, vest 0.44, **no person** |
| worker in hat + vest | hardhat 0.91, **person 0.83**, vest 0.75 | hardhat 0.78, **no person** |
| real person, no PPE | **person 0.50** | nothing |

It emits **no `Person` boxes at all** — the same defect that ruled out `baskarmother`.
Compliance is decided per person by attributing equipment to a person box, so a model
that can't find people is unusable no matter how many equipment classes it has. Adding a
"Require Safety Glasses" switch on top of it would produce a setting that either never
fires or flags nobody in particular.

**Boots** — no model surveyed has any footwear class, and the deeper problem is that
**steel-capped is not visually distinguishable from not**. A vision model can at best
report "wearing boots rather than trainers"; it cannot see the cap. That's a limit of
the sensor, not the weights, so no amount of model shopping fixes it.

What would unlock glasses: a fine-tune that has both the equipment classes *and* solid
person detection — i.e. training on something like the SH17 dataset (person, glasses,
gloves, shoes, helmet, safety-vest) rather than hoping a published PPE model has both.
Until then, hard hat and high-vis are what this app can honestly police.

<br/>

> The plate detector and the general objects model are AGPL-3.0, while this code is
> Apache-2.0. The device image carries those weights onto customer Doovits, which counts
> as distributing them, so the source for what's in the image has to stay available to
> the people running it. That is why this app lives in its own **public** repository:
> keep it public, or swap in permissively-licensed weights (and re-run the measurements
> above) before closing it. Ultralytics, who publish the objects model, also sell a
> commercial licence. The COCO vocabulary is duplicated in `common/detectors/objects.py`
> for the config schema, and `tests/test_objects.py` checks it against the weights.

<br/>

## Why not ultralytics, and why not the standard base image

**No ultralytics / torch.** ultralytics imports torch unconditionally, and there's no
torch build that fits comfortably on a Pi CM4 with ~900MB of RAM free alongside the
other app containers. The YOLO pre/post-processing is ~150 lines in `yolo.py` instead,
and is verified byte-identical to ultralytics on the same model and image (same boxes,
confidences and coordinates).

**Not built on `spaneng/doover_device_base`.** That base is now Alpine/musl. Its Python
is 3.11 at `/usr/local`, while Alpine's `py3-onnxruntime` and `py3-opencv` packages
install for Alpine's *system* Python 3.12 under `/usr/lib/python3.12` — a different
ABI, so the app's interpreter can't import them. PyPI publishes no musllinux wheels
for onnxruntime or opencv, so getting an inference stack onto that base would mean
compiling onnxruntime for musl. This app uses `python:3.11-slim-bookworm`, which has
manylinux aarch64 wheels for everything, and re-declares the labels and healthcheck
the app controller looks for.

<br/>

## Performance

Measured on a CM4 Doovit (`doovit-2c7773`), one frame at `inference_size=640`:

| | Time | Notes |
|---|---|---|
| Both models load | 676 ms | once, at startup |
| PPE per frame | ~1.4 s | 1.3–1.5 s, largely independent of source resolution |
| ANPR per frame | ~1.2 s | includes the OCR pass on each plate found |

So ~2.6 s per frame with both detectors on. The general objects model **hasn't been timed on
a CM4 yet**. On a dev laptop, single-threaded as on the device, it took 0.14 s a frame
against the PPE model's 0.27 s, so expect it to cost no more than the PPE model's ~1.4 s
on a Doovit — measure before relying on that. Snapshots are minutes to hours apart, so
latency is a non-issue.

Memory, also measured on-device: 128MB with PPE and ANPR loaded, 204MB after the first
1080p analysis, **254MB peak** — and flat at 254MB from the second run through 30
consecutive runs, so nothing accumulates.

**That's per install, and there's one install per camera.** A Doovit has ~650MB free, so
three installs with everything on could peak past it — and because `mem_limit` isn't
enforced (below), running out takes the camera apps down with it. Only enabled
detectors load, so enable just what each camera needs: a gate camera doing plates only
loads the plate models. The objects model's memory hasn't been measured on a CM4 yet.
Each install also runs its models independently, so cameras that snapshot at the same
moment (a shared schedule) compete for the same four cores. Where that's a problem, the
cloud variant has no such limit.

> The compose template's `mem_limit` is **not enforced** on current Doovits. cgroup v2
> is mounted but the memory controller isn't enabled (`cgroup.controllers` reads
> `cpuset cpu io pids`), so docker discards the limit with a warning. It's left in as
> documentation of the expected ceiling and starts working if a device ever boots with
> `cgroup_enable=memory` — but it is not protecting the camera apps today.

<br/>

## Need Help?

- Email: support@doover.com
- [Community Forum](https://doover.com/community)
- [Full Documentation](https://docs.doover.com)

<br/>

## License

This app is licensed under the [Apache License 2.0](LICENSE). Note the model weight
licences in the table above.
