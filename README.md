# pixtreme.train.common

`pixtreme-train` provides `pixtreme.train.common`: model-independent training,
image and video data, GPU preprocessing, evaluation, checkpoints, and inference
weight export. Trainer modules register their networks and model-specific steps,
then combine these components through YAML or Python.

The pure Python wheel owns only `pixtreme/train/common/` and its distribution
metadata. [pixtreme](https://github.com/sync-dev-org/pixtreme) owns
`pixtreme/__init__.py`; trainer modules own their own
`pixtreme/train/<module>/` subtrees. They can share one Python environment.

## Installation and execution requirements

Use Linux x86_64, including WSL2, with CPython 3.12 or later:

```console
pip install pixtreme-train
# Also install the converter when exporting or calibrating inference weights:
pip install 'pixtreme-train[export]'
```

For a uv project, use `uv add pixtreme-train` or
`uv add 'pixtreme-train[export]'`.

The declared runtime dependencies are `cupy-cuda13x[ctk]>=14.1`, NumPy,
`Pillow>=11`, `pixtreme>=2.24.0`, `pixtreme-transport-nvdec>=0.11.4`,
`pixtreme-transport-nvenc>=0.11.0`, `PyYAML>=6`, and `torch>=2.11`.
The `export` extra adds `tensoracle[convert]>=2.6.1`.

| Operation | Execution conditions |
| --- | --- |
| Read/write YAML, `Option`, `plain`, `resolve_config` | PyYAML; resolving a data type or an `imports` module also loads that component's Python dependencies. Built-in image data resolution loads torch and pixtreme for types and color validation, without CUDA initialization or allocation. |
| Image preprocessing, standard training and evaluation, PSNR/SSIM/VMAF | NVIDIA GPU, compatible driver and CUDA-enabled pixtreme/torch. Current pixtreme wheels require CUDA 13.x, driver R580 or later, and compute capability 7.5 or later. The selected torch wheel and its CUDA runtime must also match the driver. |
| Video indexing and packet demux | The `ffprobe` command on `PATH`, with support for the source container and codec. |
| Video source decoding | `pixtreme-transport-nvdec` and a GPU/driver supporting the actual codec, chroma layout, bit depth and dimensions. Unsupported sources are rejected; source decoding has no CPU fallback. |
| `temporal_compression`, `encode_hevc` | `pixtreme-transport-nvenc`, HEVC NVENC support (NVENC API 13.0, driver R580 or later), and NVDEC support for the selected encoded format. Source and compressed LQ decoding use the same process. |
| `export_weights`, `calibrate` | The `export` extra. Export restores weights on the CPU; calibration also requires the GPU evaluation conditions above. |

The library checks missing CUDA support and video codec capabilities before the
first update or evaluation. Codec diagnostics identify the material or selected
format and dimensions. Reading and validating configuration does not initialize
CUDA, allocate GPU memory, open source material, or start an encoder/decoder.
User-supplied `imports` modules must follow the same rule to preserve this property.

Video pixels stay on the GPU for conversion, resizing and temporal compression.
Compressed packets and bitstreams pass through host memory. An `ffmpeg` encoder
command is not part of the training path.

## Train an image model

Save this module as `my_trainer.py` on Python's import path:

```python
import torch

from pixtreme.train.common import register


@register("network", "tiny")
class Tiny(torch.nn.Module):
    def __init__(self, *, channels: int = 3) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + self.body(image)
```

Save the configuration as `config.yaml`, replacing the image directory with your
material directory:

```yaml
imports: [my_trainer]
networks:
  main: {type: tiny}
losses:
  pixel: {type: l1, weight: 1.0}
optimizers:
  main: {type: adamw, networks: [main], lr: 2.0e-4}
data:
  train:
    type: image_degradation
    directory: /data/train
    patch_size: 64
    degradation:
      name: identity
      seed: 0
      gt: {steps: [crop]}
      lq: {steps: [crop]}
train:
  iterations: 1000
  seed: 0
  device: cuda
  checkpoint_interval: 500
  batch_size: 8
  prefetch_batches: 2
  sample_threads: 1
```

```console
pixtreme-train train config.yaml --output runs/first
```

`train(config, output, *, resume=None)` runs the same operation from Python and
returns the last checkpoint path. The output must be a new directory. A run
writes `config.yaml` with resolved defaults, `environment.json`, numbered JSONL
segments under `records/`, the completed run's `training.jsonl`, and
`checkpoint_XXXXXXXX.pt` files. Updates record their loss/metrics and learning
rates. Evaluations add reports under the run directory.

### Common configuration and training parts

Top-level keys are `imports`, `networks`, `losses`, `optimizers`, `schedulers`,
`ema`, `step`, `data`, `train`, and `notes`. Data-specific keys belong inside the
chosen `data.train` or `data.evaluation.<name>` entry. `resolve_config(mapping)`
checks these keys, asks each registered data type to validate its own entry, and
returns a new mapping with defaults. `write_yaml(path, mapping)` saves it.

The default step is `{type: default}`: one network named `main`, one optimizer,
and at least one weighted supervised loss. Built-in losses are `l1`, `mse`, and
`charbonnier`; optimizers are `adam`, `adamw`, and `sgd`. Their remaining keys are
passed to the selected component. `schedulers.<optimizer-name>` can select
`{type: warmup_cosine, warmup: 100, minimum: 0.0}`; the runner supplies the total
iteration count. `ema: {decay: 0.999}` enables averaged evaluation weights.

| `train` key | Behavior/default |
| --- | --- |
| `iterations`, `seed`, `device`, `checkpoint_interval` | Required. Iteration and checkpoint intervals are positive; seed is nonnegative. |
| `precision` | `bf16` by default, or `fp32`. Parameters and optimizer state remain float32. |
| `channels_last` | Boolean, default `false`; stores all networks' 4D weights and 4D tensor inputs in channels_last order, including EMA evaluation inputs. Shapes and values are unchanged. Video tensors with five or more dimensions keep their existing layout. |
| `cudnn_benchmark` | Boolean, default `false`; enables cuDNN convolution algorithm benchmarking during training and restores the caller's flag on success or failure. When false, the caller's flag is left alone. |
| `compile` | `null` by default, or a [torch.compile mode](https://docs.pytorch.org/docs/stable/generated/torch.compile.html): `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. Compiles each registered network's forward; checkpoint, EMA and exported weight names stay unchanged. Initial compilation takes time and may use additional memory. |
| `batch_size` | Positive integer, default `1`. |
| `prefetch_batches` | Nonnegative integer, default `0`; bounds queued and in-progress batches together, including batches whose sample reads are pending. Zero creates only the batch being requested. Grouped video reads may also retain the unused tail of one clip, as described below. |
| `sample_threads` | Positive integer, default `1`; builds samples in threads in the training process, preserving sampler order. With prefetch, reads from different batches share the same pool within the prefetch bound for datasets without `state_dict` and for standard video identity state. Custom mutable state keeps reads parallel within each batch. |
| `evaluation_interval` | Required and positive when `data.evaluation` is nonempty. |

`register(kind, name)(component)` and `lookup(kind, name)` support `network`,
`loss`, `step`, `optimizer`, `scheduler`, `dataset`, and `degradation`. Unknown
names report the available names. Extend `TrainingStep`, or `DefaultStep` for a
compatible supervised model, to own model-specific updates, state or evaluation.
Network classes and steps can implement `check_config`; steps can implement
`check_resume`. A custom dataset receives `seed` and `device`, resolves its own
configuration through an optional `resolve_config` method, and can provide paired
`state_dict`/`load_state_dict` methods and `close` for lifecycle management.

A custom dataset used with `sample_threads > 1` must support concurrent
`__getitem__` calls, protect shared mutable state and derive random choices from
the seed and sample number. Its checkpoint state must be independent of sample
completion order. Datasets supplying `batch_sampler` continue to return complete
batches sequentially. Cross-batch reads are enabled only when the dataset has no
`state_dict`, or its `state_dict` is the standard `VideoDataset` implementation
(including subclasses that inherit it unchanged). Other datasets with
`state_dict` finish each batch before submitting the next, so their checkpoint
state excludes unused future samples. A sample failure retains its exception
type and adds the batch and sample numbers through the exception's `__notes__`;
the CLI prints those notes.

### Image data and degradation

`image_degradation` requires `patch_size`, `degradation`, and exactly one of
`paths` or `directory`. Optional `prefixes` filter direct child filenames;
`augment` defaults to `true`. Supported image suffixes are PNG, JPEG, BMP, TIFF,
WebP and EXR. Training samples include CHW float32 `gt`/`lq` tensors, `path`,
and the global sample `number`; evaluation samples include the material `index`.

`image_evaluation` uses the same source and degradation description without
`patch_size` or augmentation. It uses full images, trimming the bottom and right
for degradation divisibility and ignoring crop markers. `image_pair_evaluation`
accepts `pairs: [[input_path, reference_path], ...]`, `scale` (default `1`),
optional `input_color`/`reference_color` declarations, and `output_color`
(default `{colorspace: Rec.709, gamma: sRGB}`). It checks paired dimensions and
uses the corresponding full region.

Each source color attribute is resolved from file metadata, the optional
`source_color` declaration, then the format default: sRGB/sRGB for raster images
and ACES2065-1/linear for EXR. Conflicting or uninterpretable declarations fail
before training. Preprocessing starts in ACES2065-1/linear on the GPU and passes
tensors to torch through DLPack.

`DegradationPipeline.from_mapping(mapping)` and `load_pipeline(path)` validate
named `gt.steps` and `lq.steps`; `to_mapping()` returns the resolved description.
Steps include `color`, `resize`, `compression_artifacts`, the registered blur and
noise operations, and a value-free `crop` marker. `compression_artifacts` models
intraframe compression through pixtreme; temporal HEVC belongs in the video data
configuration. A custom degradation operation supplies `validate(parameters,
location)` and `apply(frame, values, *, stream)`, with optional `scale` and `unit`
geometry attributes.

A numeric pair denotes an inclusive range; `{"choices": [...]}` selects disjoint
candidates, and string lists select names. All candidates are validated.
`resize` runs in linear light. A `color` step declares its `colorspace` and
`gamma`; a sequence without a final color step receives Rec.709/sRGB.
Training requires one crop marker in each branch; `patch_size` is the final GT
size, and crop/augmentation preserve correspondence with LQ. Random choices are
derived from the seed and global sample number, including with thread prefetch.

`value_clipping` is a **data** key, default `codec_only`: preserve negative and
above-one scene values except at codec boundaries. `color_and_metrics` also clips
after color steps and clips prediction/reference during standard evaluation.
Final color and clipping choices are saved in the configuration and checkpoint.

For direct use, `DegradationPipeline.apply(frame, generator, *, patch_size=None,
crop_seed=0, number=0, value_clipping="codec_only")` accepts a GPU pixtreme `Frame`
with **float32 RGB pixels, ACES2065-1 primaries and linear transfer**, and returns
`{"gt": Frame, "lq": Frame}`. A Rec.709/sRGB Frame must be converted first;
changing its metadata alone does not convert its pixels.
`ImageDegradationDataset(paths, pipeline, *, seed,
patch_size, ...)` and `ImageEvaluationDataset(paths, pipeline, ...)` wrap source
reading, color handling and tensor conversion.
`ImagePairEvaluationDataset(pairs, *, scale=1, input_color=None,
reference_color=None, output_color=None, device="cuda", value_clipping="codec_only",
seed=None)` wraps explicit input/reference pairs.

### Direct preprocessing with Frames

Given a GPU pixtreme Frame `image` and a validated `pipeline`, prepare its channels,
color and dtype explicitly. This also converts integer storage to float32 through
pixtreme's output-storage conversion:

```python
import numpy as np
import torch
from pixtreme.color import rgb_to_rgb

from pixtreme.train.common import rgb_frame

stream = torch.cuda.current_stream(image.__dlpack_device__()[1])
rgb = rgb_frame(image)
rgb.wait(stream)
linear = rgb_to_rgb(
    rgb,
    output_colorspace="ACES2065-1",
    output_gamma="linear",
    dtype="float32",
    stream=stream,
)
rgb.record_stream(stream, write=False)
result = pipeline.apply(linear, np.random.default_rng(0))
gt, lq = result["gt"], result["lq"]
```

For a training pipeline containing crop markers, also pass `patch_size` (the final
GT size), `crop_seed` and the global sample `number` to `apply`. The image datasets
perform source decoding and this input conversion for their callers.

## Video data and temporal compression

Choose the registered data type `video` for training or `video_evaluation` for
ordered evaluation. Supply `paths` containing video files or saved `VideoIndex`
JSON files. These datasets produce TCHW float32 GPU clips. A model-specific
training/evaluation step must handle the temporal dimension and model state;
the default image step does not adapt a video model automatically.

| Video data key | Default/meaning |
| --- | --- |
| `frames`, `scale`, `frame_step` | `1`, `1`, `1`; clip length, spatial reduction, and source-frame stride. |
| `time_range` | `null`, or `[start, end]` in seconds. |
| `gop_aligned` | `false`; constrain selected starts to GOP boundaries when enabled. |
| `scene_cut_probability` | `0.0`; mixing needs at least two frames and two distinct materials. Evaluation forbids mixing. |
| `crop_size` | `null`, or GT crop size divisible by `scale`, selected before LQ reduction and temporal compression; it is the final GT size unless `patch_size` is set. Clips can mix source dimensions when a common crop is set. |
| `augment` | `false`; evaluation requires `false`. Transpose must preserve batch dimensions. |
| `colorspace`, `gamma` | `Rec.709`, `Rec.709`; RGB output color. Source color comes from video metadata. |
| `temporal_compression` | `null`, or a HEVC settings mapping as below. |
| `patch_size` | Training only: `null` (return the entire prepared clip), or final GT patch size divisible by `scale`. Patches are cut after temporal compression. When set, it must fit `crop_size` and every source. |
| `patches_per_clip` | Training only: `1`; positive integer number of consecutive samples from each decoded/compressed clip. Values above `1` require `patch_size`. Independent of `train.batch_size`; a clip's samples may span batches. |

`crop_size` keeps its existing meaning: it selects the GT region before LQ
reduction and temporal compression. `patch_size` selects a smaller region after
those operations, then applies each sample's augmentation to every field and
frame. A patch's coordinates and size must align with every field's pixel grid.
With `patch_size: null` and `patches_per_clip: 1`, pixels, sample records, and
random choices are unchanged.

For example, to produce 16 separate GT 128 / LQ 64 samples from one prepared
512-pixel GT clip, merge these fields into your training configuration. This
example uses `train.batch_size: 16`; a smaller or nondivisible batch size works too:

```yaml
train:
  batch_size: 16
data:
  train:
    type: video
    paths: [training.mov]
    frames: 10
    scale: 2
    crop_size: 512
    patch_size: 128
    patches_per_clip: 16
    augment: true
    temporal_compression: {}
```

The 16 samples share material, frame timing, the source crop, and compression
settings. Patch positions and horizontal/vertical flips and transposition are
chosen per sample; patches may overlap. Sharing clips reduces material/time and
compression diversity within each batch. The source clip is decoded once and
LQ is compressed once per group. `train.sample_threads` runs different groups
concurrently, including groups from subsequent batches within
`train.prefetch_batches`; thread count and prefetch do not change sample choices. Numbered
sample `n` belongs to clip `n // patches_per_clip` and patch
`n % patches_per_clip`. The group's first material draw chooses its clip; the
runner consumes the other material draws too, preserving sampler progress.
The runner takes consecutive samples from this sequence to fill each batch,
retaining the rest of a clip for the following batch. For example, batch 16
with `patches_per_clip: 32`, or batch 10 with `patches_per_clip: 20`, splits
one clip across two batches. Batch 8 with `patches_per_clip: 12` places the
next clip's boundary inside the second batch. Each clip is decoded and
compressed once during uninterrupted training.

The number of created or in-progress samples not yet delivered in a batch is
bounded by `max(1, train.prefetch_batches) * train.batch_size + patches_per_clip - 1`.
The extra `patches_per_clip - 1` allows the unused tail of one clip; sample
threads, pending reads and retained patches are reclaimed on every exit.
Resume rebuilds the clip containing the first unused patch and any later
prefetched clips, skipping patches already used by completed updates. Inside
a clip, checkpoint `rng.generators.sampler` records the generator state before
that clip's first material draw; `iteration * train.batch_size % patches_per_clip`
identifies the used patch count. At a clip boundary, that generator state is
before the next clip's first draw. The checkpoint format and video data state
version are unchanged. Resume requires the same batch size and patch count.

When `patch_size` is set, sample records additionally contain `clip_number`,
`patch_number`, `patch_crop` (top, left, height, width in the prepared GT clip),
`source_crops` (one top/left/height/width source region per frame), and
`augmentation` (`hflip`, `vflip`, `transpose`, applied in that order after crop).
`patch_crop` uses coordinates before augmentation; `source_crops` locates the
prepared clip in the source frames. GT-to-LQ coordinates follow `scale`.
Direct `VideoDataset[(number, material)]` access with multiple patches uses
`material` as that clip's material; use the same material for its other patch
numbers. The runner groups these reads automatically. Evaluation data has no
`patch_size` or `patches_per_clip` option.

Samples carry `paths`, `frame_numbers`, `times` (float seconds), `pts`,
`time_base` (numerator/denominator pairs), per-frame `reset` flags, and
`continuous`, as well as `gt`/`lq` and the sample `number`. Exact source times are
`Fraction(pts, 1) * Fraction(*time_base)` for each frame. Evaluation also supplies
`path`, `index`, and `color`. Saved indices are checked against source identity;
changed source material requires rebuilding the index.

For example, set the video entry's `temporal_compression` to
`{rate_control: cbr, bitrate: [300000, 700000], gop_length: 10, b_frames: 0}`.
Settings are selected once per clip from the sample's generator; `sample["compression"]`
records the chosen settings. All selectable codec formats are checked before use.
Numeric settings accept a scalar, an inclusive range (`[min, max]` or
`{range: [min, max]}`), or discrete `{choices: [...]}`.

| Temporal compression key | Default/accepted values |
| --- | --- |
| `codec`, `gop_length`, `b_frames` | `hevc`, `10`, `0`. |
| `rate_control` | `cbr`, `vbr`, or `constqp`; default `cbr`. |
| `bitrate`, `max_bitrate`, `qp` | `500000` bit/s, selected bitrate, `24`; QP is used for constant-QP control. Every selected maximum bitrate must cover every bitrate candidate. |
| `references` | `null`: encoder default; otherwise an integer from `1` to `7`, selecting the forward reference count (NVENC `num_ref_l0`, corresponding to x265's `ref`). |
| `vbv_buffer_size` | `null`: encoder default; otherwise an integer from `1` to `4294967295` bits, corresponding to FFmpeg's `-bufsize`. VBV initial delay stays at the encoder default. |
| `subsampling`, `bit_depth` | `4:2:0` and `8`; also `4:2:2`/`4:4:4` and `10` where both GPU codecs support them. Use `choices` for disjoint bit depths. |
| `matrix`, `range` | `BT.709`, `legal`; matrices also accept `BT.601`/`BT.2020`, ranges also accept `full`. |
| `frame_rate` | `null`: source average display rate divided by `frame_step`; a positive rational such as `"30000/1001"` overrides it. One-frame sources require an explicit rate. |
| `motion_vectors`, `residuals`, `frame_types` | `false`; request `mv`, `residual`, `frame_type` GPU fields. |

Omitting `references` or `vbv_buffer_size`, or setting either to `null`, preserves
the encoder's defaults. `choices` must contain integers; it cannot mix `null`
with explicit values. Explicit `0` is rejected in configuration. In the sample's
`compression` record, `0` for either control means that the encoder default was
used; this numeric representation lets the runner collate samples into a batch.
`resolve_config` and `select` keep these defaults as `None`. With an explicit
reference count, the DPB holds that many forward references plus one backward
reference when B frames are enabled (`num_ref_l1=1`); without B frames the DPB
holds only the forward references. Reference capability is checked before use.

For settings similar to x265's `ref=1` and a VBV buffer of twice the bitrate,
add this mapping inside the video data entry:

```yaml
temporal_compression:
  rate_control: cbr
  bitrate: 500000
  references: 1
  vbv_buffer_size: 1000000
  b_frames: 0
```

`vbv_buffer_size` is a fixed number of bits, not a multiplier of the selected
bitrate. If bitrate varies per clip, choose the buffer independently; this example
has a two-second buffer only at 500000 bit/s. NVENC and x265 need not produce the
same pixels or bitstream with similar controls.

Motion vectors are displacement fields; geometric
transforms adjust their direction and magnitude. Frame type codes are B=`0`,
P=`1`, I=`2`. `transform_video_fields` applies corresponding spatial transforms
to image, vector, scalar and frame fields.

Temporal compression enables NVDEC analysis only when `motion_vectors` or
`residuals` is requested. Pictures and frame types alone skip that work. Without
analysis, `faulty_reference` and `missing_slices` metadata are unavailable;
missing references and native decoder errors still raise `RuntimeError`.
This applies to clips encoded internally by temporal compression. Source video
readers also use the decoder's default mode without analysis.

Direct Python operations include `VideoIndex.build/save/load`, `VideoFrame`,
`ClipSampler`/`ClipSelection`, `VideoField`/`VideoSample`,
`VideoClipReader.read/close`, and `VideoDataset`/`VideoEvaluationDataset`.
`TemporalCompression.resolve_config` validates settings without running codecs;
`select(generator)` returns scalar settings, `validate(...)` checks capabilities,
and `apply(lq, generator, *, frame_rate, ...)` returns fields and chosen settings.
`encode_hevc(lq, parameters, *, frame_rate, colorspace="Rec.709", gamma="Rec.709")`
encodes a nonempty TCHW floating CUDA RGB tensor to Annex B HEVC bytes. Pass
settings from `TemporalCompression(...).select(generator)`.
`decode_hevc(source, *, device=0, motion_vectors=False, residuals=False,
frame_types=False)` accepts HEVC bytes or a file path and returns `DecodedLQFrame`
objects in display order; `device` is the CUDA device index.
This standalone operation always enables analysis and raises `RuntimeError` for
missing or faulty references and missing slices, even when no byproducts are requested.

Call `close()` on a directly constructed video reader or dataset when finished.
The training runner closes data resources on completion, error and signal stop.
Readers reuse bounded per-thread decoder sessions; samples retain valid pixel
values after subsequent reads and after readers close.

## Evaluation and inference weights

`measure(index, reference, candidate)` measures RGB float32 GPU Frames using
pixtreme PSNR and SSIM with data range `1.0`. Both Frames must have matching
shape, color and device and dimensions of at least 11 pixels. It does not change
color, crop, normalize or clip. Use `rgb_frame(frame)` explicitly to obtain RGB
channels from a supported pixtreme Frame, then perform any required color or
dtype conversion yourself. `FrameMetrics`, `MaterialMetrics` and
`EvaluationMetrics` preserve per-frame results and aggregate by frame count;
`json_metric` represents positive infinity as `"Infinity"` for strict JSON and
rejects NaN/negative infinity.

`measure` requires a CUDA device and stream; CPU-only tests that substitute
Frames and metrics must substitute `measure` itself.

`evaluate_material(path, references, candidates, *, conditions, correspondence,
video=False)` adds provenance through `EvaluationConditions` (region,
preprocessing, network, weights, precision, colorspace, gamma, normalization,
value_clipping, and data_range=`1.0`) and `FrameCorrespondence` (reference and
candidate indices and optional times in seconds). Video requires strictly
increasing indices and finite increasing times on **both** timelines, even for
one frame; timeline offsets can differ. Callers prepare corresponding RGB Frames
before evaluation. `measure_vmaf(..., conditions=..., correspondence=...,
model="vmaf_v0.6.1")` returns `VmafMetrics` using pixtreme's VMAF implementation
(minimum dimensions 33 pixels). `write_evaluation(path, result)` atomically writes
a complete strict-JSON report.

The runner passes `context` to `TrainingStep.evaluate(datasets, output, *, context)`.
Direct calls must supply mappings `networks` (configuration by network name),
`weights` (checkpoint/version descriptions by network name) and `datasets`
(preprocessing descriptions by dataset name). Optional `run` and
`completed_updates` identify a training call. Overrides must accept this keyword
and record equivalent provenance. Standard evaluation prefers EMA when present,
runs in float32, and restores the network's previous training mode.

`load_network(checkpoint, name="main", *, ema=True, device="cpu")` restores a
named network in eval mode, preferring EMA and falling back to training weights
when that network has no EMA. Import the trainer's registration module first.
The checkpoint's configuration describes its data, colors and clipping policy.

`export_weights(checkpoint, output, *, name="main", ema=True, metadata,
transform=None)` writes a new artifact directory containing `weights.safetensors`
and `manifest.json`, with source checkpoint SHA256, selected network/weights and
completed update count. `metadata` must declare positive integer `scale`,
`input_color` and `output_color` (`colorspace` and `gamma`), `normalization`, and
`value_clipping`. Optional `transform(state_dict)` converts tensor names/values
before tensoracle writes safetensors. The artifact contains inference weights,
not optimizer, RNG or resumable training state.

`calibrate(checkpoint, output, *, materials, preprocess, preprocessing, targets,
infer_low_precision, metadata, ...)` also writes scales, provenance, and
`comparison.json` comparing matched float32 and model-supplied low-precision
inference. `CalibrationTarget(module, precision, input_scale, weight_scale,
application, weight="weight")` identifies a module input/weight, scale keys and
its inference operation. Precision is `int8` or `fp8` (e4m3fn), with one positive
float32 scale per tensor. `preprocess(material)` returns CHW GPU `lq`/`gt`, a
material `path`, and for video `kind="video"` plus `FrameCorrespondence` with both
times. Prepared samples stay on the GPU for comparison, so bound the material
set to available memory. The trainer owns both preprocessing and the callable
`infer_low_precision(network, sample, scales)`; calibration does not build a
TensorRT engine.

## Checkpoints, interruption and compatibility

Checkpoints use format `version: 2`, containing completed updates, network and EMA
weights, optimizer/scheduler state, model state, and committed data/sampler/RNG
state. `read_checkpoint(path)` validates this format; `save_checkpoint` and the
file helpers `write_text`/`staging_directory` publish complete artifacts atomically.
The following helpers are also public for custom training steps:

| Operation | Usage and behavior |
| --- | --- |
| `Option(name, parse, default, kinds, description, choices=(), repeat=False, required=False)` | `yaml_value(value)` validates accepted types, parses values, and applies choices; repeated options require lists. `plain(value)` converts paths and nested mappings/sequences to YAML-compatible values. |
| `save_checkpoint(path, payload, *, prefix=".checkpoint-")` | Atomically saves the caller's torch-compatible mapping; the caller assembles and validates its contents. `read_checkpoint(path)` checks a version 2 training checkpoint. |
| `write_text(path, text, *, prefix)` | Atomically writes UTF-8 text. `with staging_directory(output, prefix=...) as temporary:` publishes a new directory only after the block succeeds. |
| `rng_state(*, device=None, generators=None)` / `restore_rng_state(state, *, device=None, generators=None)` | Capture/restore Python, NumPy, torch CPU, optional CUDA and named torch generators; restoration validates all streams first. |
| `NumberedSampler(size, generator, *, start=0)` | Yields `(global_sample_number, random_material_index)` with replacement. `sample_generator(seed, number)` supplies that sample's NumPy generator. `EndlessSampler(size, generator)` draws uniform material indices with replacement from the supplied torch generator. |
| `image_paths(directory, *, prefixes=())` | Lists supported direct child images in filename order, rejecting an empty selection. `augment_images(images, *, hflip, vflip, transpose)` applies matching spatial transforms to a mapping of tensors. |
| `warmup_cosine_factor(completed, *, warmup, total, minimum)` | Returns the linear-warmup/cosine-decay learning rate coefficient. |
| `update_ema(average, model, *, decay)` | Averages floating state and copies nonfloating state. `check_finite_loss(loss, *, iteration)` and `check_finite_gradients(parameters)` reject nonfinite values; finite gradients are not clipped. |
| `release_gpu_memory(device)` | Returns unused torch and CuPy allocator blocks for a CUDA device. |
| `training_signals()` | Context manager exposing the first interruption to the training loop and escalating a second signal. `worker_signals(worker_id)` configures a DataLoader worker; `stop_loader_workers(iterator)` requests shutdown, waits up to five seconds and reaps survivors. |

```console
pixtreme-train train config.yaml --output runs/continued --resume runs/first/checkpoint_00000500.pt
```

Resume permits changes to `train.iterations` (must exceed completed updates),
`train.device`, `train.sample_threads`, `train.channels_last`,
`train.cudnn_benchmark`, `train.compile`, and `notes`. Other settings must match the
saved configuration, including data sources, degradation, color, batch size and
prefetch depth. Version 2 checkpoints lacking `sample_threads` use `1`; missing
network execution options mean `channels_last: false`, `cudnn_benchmark: false`,
and `compile: null`. These options may change on resume in either direction
without converting checkpoint weights. Model
steps may further restrict resumption. The next sample follows the last completed
update; unused prefetched samples are regenerated.

The same sources, configuration, environment and seed reproduce material/sample
selection and preprocessing random choices. Bitwise identity of model updates
across GPUs, library versions or arithmetic modes is not promised. Codecs can
also depend on hardware/driver behavior; compare numerical outputs under matched
conditions and a tolerance suitable for the operation.

The first SIGINT/SIGTERM stops new batch production, finishes an in-progress
update, and saves the last completed state. No checkpoint is written before the
first completed update or when no update has advanced since the last checkpoint.
An interrupted evaluation does not become a successful report. A second signal
exits without waiting for saving.

### Migration from 0.3.x

Use the following replacements when updating trainer modules and configurations:

| 0.3.x usage | Current usage |
| --- | --- |
| `data.train.batch_size`, `data.train.prefetch_batches` or `num_workers` | Move batch size and prefetch depth to `train.batch_size` and `train.prefetch_batches`; use `train.sample_threads` for concurrent sample creation in one process. |
| Top-level `degradation`/`evaluation_degradations`; `train.value_clipping` | Put each pipeline and `value_clipping` inside its data entry. Split the former `{name, seed, steps}` pipeline into `{name, seed, gt: {steps}, lq: {steps}}`; the example below shows training and named evaluations. |
| `data.evaluation: {type: image_evaluation, ...}` | Create one `data.evaluation.<name>` entry for the training pipeline and one for each former `evaluation_degradations` pipeline; explicitly give each entry its sources and degradation. |
| `data.train.region_size` or `ImageDegradationDataset(..., region_size=..., patch_size=...)` | Remove `region_size` from YAML and Python. Keep `patch_size` as the final GT size; place one `crop` marker in each training branch. |
| NumPy input and one LQ ndarray returned by `DegradationPipeline.apply` | Supply float32 RGB on the GPU in ACES2065-1/linear, as in [the Frame example](#direct-preprocessing-with-frames). The result is a mapping: use `result["lq"]` for the former LQ result and `result["gt"]` for its corresponding reference. |
| torch tensor input to `rgb_frame` | 0.3.1 treated tensor values as Rec.709 primaries and Rec.709 transfer. Explicitly upload the CHW RGB CPU float tensor and construct a GPU `Frame` with `pixtreme.io.from_tensor(..., layout="CHW", colorspace="Rec.709", gamma="Rec.709", channels="RGB")`, as in [the example below](#migrating-tensor-inputs-to-rgb_frame). Different color metadata changes metric values; `rgb_frame` prepares RGB channels without changing primaries or transfer. |
| `TrainingStep.evaluate(datasets, output)` | Accept `*, context` in overrides; direct calls supply network, weight and preprocessing descriptions. |
| Implicit RGB conversion in `measure` | Prepare matching RGB float32 Frames explicitly; call `rgb_frame(frame)` for channel conversion before `measure`. `measure` requires a CUDA device and stream; CPU-only tests that substitute Frames and metrics must substitute `measure` itself. |
| `encode` steps selecting `jpeg`, `prores_lt`, `h264` or `hevc` | Replace the removed operation explicitly using the codec table below; moving `encode` into `lq.steps` still fails. |
| `VideoDataset` sample `times` as Fractions | Times are float seconds. Reconstruct exact times from each frame's `pts` and `time_base`. Video registry names are `video` and `video_evaluation`. |
| Weight export assumed available from the base install | Install `pixtreme-train[export]` and call `export_weights` with explicit inference metadata. |
| A checkpoint with `version: 1` from 0.3.x | Resume and weight extraction reject it and identify pixtreme-train 0.3.x as its supported version. There is no conversion/import operation; retain a separate 0.3.x environment to use that checkpoint. Start a new run with the current configuration. |

#### Migrating tensor inputs to rgb_frame

Given a CHW RGB CPU float32 `tensor`, this preserves the tensor's values and the
Rec.709 primaries and transfer used by 0.3.1. Transfer host data to CUDA explicitly
before constructing the Frame:

```python
import torch
from pixtreme.io import from_tensor
from pixtreme.train.common import rgb_frame

stream = torch.cuda.current_stream()
gpu_tensor = tensor.to("cuda")
gpu_tensor.record_stream(stream)
frame = from_tensor(
    gpu_tensor, layout="CHW", colorspace="Rec.709", gamma="Rec.709",
    channels="RGB", copy=True, stream=stream,
)
rgb = rgb_frame(frame)
```

Setting `gamma="sRGB"` or different primaries describes a different signal and
can change PSNR and SSIM. Choose metadata that matches the values being measured.

#### Before: 0.3.1 image training and evaluation

This configuration uses the `my_trainer` module above. In 0.3.1 the runner applied
the single `degradation.steps` list to LQ, kept the original GT region, and cropped
the corresponding pair afterward. One evaluation source entry was reused for the
training pipeline (`jpeg`) and each additional evaluation pipeline (`clean`):

```yaml
imports: [my_trainer]
networks:
  main: {type: tiny}
losses:
  pixel: {type: l1}
optimizers:
  main: {type: adamw, networks: [main], lr: 2.0e-4}
degradation:
  name: jpeg
  seed: 0
  steps:
    - encode: {codec: jpeg, quality: 80, chroma: "4:2:0", bit_depth: 8}
evaluation_degradations:
  - {name: clean, seed: 0, steps: []}
data:
  train:
    type: image_degradation
    directory: /data/train
    region_size: 128
    patch_size: 64
    batch_size: 8
    num_workers: 2
  evaluation: {type: image_evaluation, directory: /data/validation}
train:
  iterations: 1000
  seed: 0
  device: cuda
  checkpoint_interval: 500
  evaluation_interval: 100
```

#### After: current image training and evaluation

Keep reference preparation in `gt.steps` and put the former LQ-only operations
in `lq.steps`. Each training branch needs one `crop`; place it after the LQ
operations to degrade the larger image before cropping, or before them to degrade
only the patch. The example chooses patch-only artifact simulation, so its context
and edge effects differ from the old region-then-crop JPEG operation. For a
resolution-reduction task, put the reduction in LQ and use a network with the
matching output scale; the GT `patch_size` determines the corresponding LQ size.

The YAML anchor reuses the training degradation for the named `jpeg` evaluation;
evaluation ignores its crop markers and uses the full corresponding image.
The `clean` evaluation explicitly gets its own pipeline. `region_size` is removed.
Old `num_workers` counted worker processes; `sample_threads` now selects threads
within the training process, while `prefetch_batches` independently bounds batches
being prepared or queued. These are separate execution choices, not equivalent
worker-count settings.

```yaml
imports: [my_trainer]
networks:
  main: {type: tiny}
losses:
  pixel: {type: l1}
optimizers:
  main: {type: adamw, networks: [main], lr: 2.0e-4}
data:
  train:
    type: image_degradation
    directory: /data/train
    patch_size: 64
    degradation: &jpeg
      name: jpeg
      seed: 0
      gt: {steps: [crop]}
      lq:
        steps:
          - crop
          - compression_artifacts:
              quality: 80
              subsampling: "4:2:0"
              bit_depth: 8
              matrix: BT.601
              range: full
  evaluation:
    jpeg:
      type: image_evaluation
      directory: /data/validation
      degradation: *jpeg
    clean:
      type: image_evaluation
      directory: /data/validation
      degradation:
        name: clean
        seed: 0
        gt: {steps: []}
        lq: {steps: []}
train:
  iterations: 1000
  seed: 0
  device: cuda
  checkpoint_interval: 500
  evaluation_interval: 100
  batch_size: 8
  sample_threads: 2
  prefetch_batches: 2
```

The old flat `steps` mapping and every old `encode` step are rejected. None of the
following substitutions promises the old encoder's pixels, quality scale or random
sequence. Source color handling, linear-light processing and crop context also
change the data; start a new run rather than resume a version 1 checkpoint.

| Removed `encode.codec` | Replacement and change in degradation meaning |
| --- | --- |
| `jpeg` | In `lq.steps`, use `compression_artifacts` with explicit `quality`, `subsampling` (formerly `chroma`), `bit_depth`, matrix and range. This GPU operation simulates compression artifacts; it does not run the former Pillow JPEG bitstream encode/decode and is not a pixel-exact substitute. |
| `prores_lt` | For image-only artifacts choose `compression_artifacts` and select its settings explicitly. There is no ProRes LT encoder or ProRes-quality equivalent in the standard degradation pipeline. |
| `h264` | For image-only artifacts choose `compression_artifacts`. For a temporal encoding task, use video `temporal_compression`, which encodes HEVC; it changes codec and does not reproduce the former H.264 encoder or QP behavior. |
| `hevc` | For actual clip encoding choose video `temporal_compression` with NVENC/NVDEC and explicitly selected rate-control/GOP settings. This is clip-based hardware HEVC rather than the former FFmpeg still-image encode. For image-only simulated artifacts choose `compression_artifacts`. |

Version 2 training checkpoints and inference artifacts are distinct formats.
Inference artifacts cannot be used as resume checkpoints. The public API is
`pixtreme.train.common` and the documented registry contracts; private modules
and underscore-prefixed helpers are not compatibility interfaces.

## License

MIT. See [LICENSE](LICENSE).
