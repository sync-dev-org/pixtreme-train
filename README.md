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
| Compressed evaluation LQ export (`stream(..., lq_path=...)`) | The `ffmpeg` command on `PATH`, supporting HEVC in NUT and MP4 stream copy. This command is required only when export is requested. |
| `export_weights`, `calibrate` | The `export` extra. Export restores weights on the CPU; calibration also requires the GPU evaluation conditions above. |

The library checks missing CUDA support and video codec capabilities before the
first update or evaluation. Codec diagnostics identify the material or selected
format and dimensions. Reading and validating configuration does not initialize
CUDA, allocate GPU memory, open source material, or start an encoder/decoder.
User-supplied `imports` modules must follow the same rule to preserve this property.

Video pixels stay on the GPU for conversion, resizing and temporal compression.
Compressed packets and bitstreams pass through host memory. An `ffmpeg` encoder
command is not part of the training path.
The run's `environment.json` records both `pixtreme-transport-nvenc` and
`pixtreme-transport-nvdec` package versions alongside pixtreme and torch.

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
segments under `records/`, a consolidated `training.jsonl` when the run
completes or stops on the first signal (not when it fails), and
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

To decay from the initial learning rate without warmup, select
`{type: warmup_cosine, warmup: 0, minimum: 0.2}`. The coefficient starts at `1.0`
before any update and reaches `minimum` after the total number of updates;
positive warmup lengths retain linear warmup followed by cosine decay.
The warmup length must be nonnegative and less than the total iteration count,
and `minimum` must be between `0` and `1`, inclusive. Invalid settings are
rejected before training starts. The scheduler counts completed updates and
continues from the saved count when resuming a checkpoint.

| `train` key | Behavior/default |
| --- | --- |
| `iterations`, `device`, `checkpoint_interval` | Required. Iteration and checkpoint intervals are positive. |
| `seed` | Optional nonnegative integer. Omit the key to choose a random seed once when starting a new run; `null` is rejected. The actual seed is saved in `config.yaml` and each checkpoint's configuration. Reuse that saved configuration to repeat the run. |
| `precision` | `bf16` by default, or `fp32`. Parameters and optimizer state remain float32. |
| `channels_last` | Boolean, default `false`; stores all networks' 4D weights and 4D tensor inputs in channels_last order, including EMA evaluation inputs. Shapes and values are unchanged. Video tensors with five or more dimensions keep their existing layout. |
| `cudnn_benchmark` | Boolean, default `false`; enables cuDNN convolution algorithm benchmarking during training and restores the caller's flag on success or failure. When false, the caller's flag is left alone. |
| `compile` | `null` by default, or a [torch.compile mode](https://docs.pytorch.org/docs/stable/generated/torch.compile.html): `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. Compiles each registered network's forward; checkpoint, EMA and exported weight names stay unchanged. Initial compilation takes time and may use additional memory. |
| `batch_size` | Positive integer, default `1`. |
| `prefetch_batches` | Nonnegative integer, default `0`; bounds queued and in-progress batches together, including batches whose sample reads are pending. Zero creates only the batch being requested. Grouped video reads may also retain the unused tail of one clip, as described below. |
| `sample_threads` | Positive integer, default `1`; builds samples in threads in the training process, preserving sampler order. With prefetch, reads from different batches share the same pool within the prefetch bound for datasets without `state_dict` and for standard video identity state. Custom mutable state keeps reads parallel within each batch. |
| `supply_timeout` | Finite seconds greater than zero, integer or float, default `600`. Bounds the wait for the next training batch or evaluation data element, and the time allowed to save after a supply timeout. Always enabled; `null`, zero and negative values are rejected. |
| `evaluation_interval` | Required and positive when `data.evaluation` is nonempty. |

`register(kind, name)(component)` and `lookup(kind, name)` support `network`,
`loss`, `step`, `optimizer`, `scheduler`, `dataset`, and `degradation`. Unknown
names report the available names. Extend `TrainingStep`, or `DefaultStep` for a
compatible supervised model, to own model-specific updates, state or evaluation.
Network classes and steps can implement `check_config`; steps can implement
`check_resume`. Steps can also override these optional, default no-op methods:

- `initialize(self)` runs once after step construction for a **fresh** training
  call, before data is supplied or any update runs. Load initial model weights
  here and initialize any EMA or model-specific averaged weights together. Load
  tensors into the existing network objects so optimizer parameters stay attached.
  Resume calls `load_state_dict` instead and never calls `initialize`.
- `check_checkpoint(cls, saved)` is a classmethod receiving the complete,
  format-checked checkpoint, including `saved["config"]` and `saved["step"]`.
  Validate model-specific records here and raise `ValueError` identifying the
  record and reason when they are missing or inconsistent. This method runs on
  resume before step construction and on network extraction, export and
  calibration before applying weights. It must not mutate `saved` or construct
  training resources. Inference never constructs the training step or optimizer.

Move fresh-only weight file reads from constructors and `check_config` into
`initialize`. A resumed run can then continue after that initial file is removed;
the resolved configuration can keep its original path. Keep configuration shape
checks in `check_config`, comparisons between the saved and requested
configurations in `check_resume(cls, saved_config, requested_config)`, and instance
state restoration in `load_state_dict(self, state)`. Implement model record checks
once in `check_checkpoint` so resume and inference share the same interpretation.

A custom dataset receives `seed` and `device`, resolves its own
configuration through an optional `resolve_config` method, and can provide paired
`state_dict`/`load_state_dict` methods and `close` for lifecycle management.

A dataset that assembles complete batches can implement
`batch_sampler(generator, completed, *, batch_size)`. The runner passes the
resolved `train.batch_size` and the number of completed updates (`0` for a new
run). The sampler yields indices for the dataset's `__getitem__`, whose result
is passed directly to the update without another collation. Random material
choices should use the supplied torch generator; derive sample numbers from
`completed` and `batch_size` so resuming continues from the completed batch.
Batch size remains in `train`; `data.train.batch_size` is rejected.

Existing `batch_sampler(generator, completed)` implementations keep their
two-argument call and batch behavior. To receive the shared size, add an explicit
`batch_size` parameter accepting a keyword (as above, or as a regular parameter).
Accepting only `**kwargs` does not opt in, and uninspectable callables retain the
two-argument call. Dataset factories and data types without a batch sampler
keep their existing inputs and behavior.

Data acquisition runs in the training process but outside the learning thread,
including with `sample_threads: 1` and `prefetch_batches: 0`. Initialize any
thread-local decoder state in the thread that performs the read. A custom
dataset used with `sample_threads > 1` must support concurrent
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

### Describe components and configuration keys

Inspect registered components without starting training, allocating GPU resources,
opening material files, or executing codecs:

```bash
pixtreme-train describe --import my_trainer > components.json
```

Repeat `--import MODULE` to load several registration modules. Python callers
use the same operation. The command sends module-import messages to standard
error so standard output contains just the JSON document:

```python
from pixtreme.train.common import describe_components

metadata = describe_components(imports=["my_trainer"])
network = metadata["components"]["network"]["my_network"]
fields = network["schema"]["properties"]
```

Use the registered name from your trainer in place of `my_network`. Reading the
descriptions imports the framework's built-in components and the explicitly
requested modules, including their dependencies such as torch. It does not
discover or import other training repositories. Registrations already loaded in
the Python process also appear. A registration module and its schema provider
must themselves avoid starting training or allocating GPU resources at import
or description time.

The result contains these JSON-compatible entries:

| Entry | Contents |
| --- | --- |
| `imports` | Explicit module names passed to this call. Include these modules in the YAML's `imports` so another process can resolve the same names. |
| `components.<kind>.<name>` | `description`, `source` (`declaration` or `signature`), `schema`, `injected`, and `unknown`. Kinds are the same seven kinds accepted by `register`. |
| `configuration` | JSON Schema describing the common YAML envelope, defaults, training options, and references between named components. |
| `degradation_pipeline` | JSON Schema for the ordered GT/LQ pipeline, from the same declaration used by its validator. |
| `temporal_compression` | JSON Schema for temporal compression, from the same declaration used by its validator. |

`configuration` marks component selections with `x-component-kind`. For example,
`networks.additionalProperties` selects a `network`, and `data.train` selects a
`dataset`. Choose its registered `type`, then use that component's `schema` for
the remaining entry. JSON Schema `$ref` values are local to each returned
schema; retain its `$defs` when rendering nested fields. Degradation operation
schemas describe the parameters under the operation name, for example
`{gaussian_blur: {sigma: 1.0}}`, rather than a `type` entry.

Loss entries include `weight` (default `1.0`), which the step uses outside the
loss factory. Optimizer entries include a required `networks` list referring to
configured network names. Scheduler entries use their optimizer's name as the
mapping key. `injected` names the arguments supplied by the framework, such as
optimizer parameters, the scheduler's optimizer and total update count,
dataset seed and device, and the step's networks, losses and common settings.
These arguments are excluded from the component's user-input schema.

Without an explicit declaration, the operation reads the factory's signature
(names, type annotations and defaults) and its Google, NumPy or reST argument
docstrings. Missing information remains explicit: `unknown.keys` is true when
the accepted keys cannot be enumerated, and `unknown.fields` maps field names to
missing `type`, `description`, or JSON-serializable `default` information.
Opaque `**kwargs` factories and custom degradation operations remain available;
`additionalProperties: true` permits their keys to be entered manually. A missing
description or type does not hide a component or require its author to add a
declaration.

A registered class, factory, or degradation operation can optionally expose
`config_schema()` with no arguments, returning an object JSON Schema for its
accepted YAML mapping. This declaration takes precedence over signature and
docstring extraction. It describes component options; the framework supplies
the selection and composition keys (`type`, loss `weight`, optimizer
`networks`). Built-in datasets and degradation operations return the same
declarations used for validation. Custom datasets should likewise return a
schema from the declaration used by their `resolve_config` method. The method
is optional and does not change registration or training for existing parts.

Descriptions guide configuration entry; they do not replace `resolve_config`
or component-specific checks. Conditions between fields may require additional
validation. In particular, leave `train.seed` absent to choose it once at run
start or reuse the checkpoint's seed on resume. There is no configuration
default for that key, and explicit `null` is invalid.

### Build configuration in a browser

Install the optional WebUI dependencies and start the local configuration builder:

```console
pip install 'pixtreme-train[webui]'
pixtreme-train webui --import my_trainer
```

Open `http://127.0.0.1:7860`. Repeat `--import MODULE` for additional registration
modules; `--port PORT` selects another port between 1 and 65535. The server binds
to `127.0.0.1`. It loads the selected modules at startup and includes the same
module list in exported YAML. Restart with different `--import` arguments to
change that list. Without the extra, the command explains how to install it;
training and `describe` do not require these dependencies.

Use **Add entry** to name networks, losses, and optimizers. Select their registered
types, connect each optimizer's `networks` list to the network names, then choose
`data.train` and enter the training progress settings. For the default step, name
the single network `main`, add at least one loss, and use one optimizer. Enable an
optional key with its **Include** checkbox; leaving it unchecked omits the key.
The displayed defaults describe what resolution supplies when a key is omitted;
the form does not insert component constructor defaults. Leave `train.seed`
unchecked to preserve its omission rather than setting it to `null`.

The form handles nested objects, named mappings, arrays and fixed-length tuples,
enum and constant choices, `anyOf`/`oneOf` alternatives, and local JSON Pointer
`$ref` references. Select the `choices` or `range` alternative to enter sampled
values. For opaque component keys, use **Add entry** and enter the value as JSON.
Fields with unknown types, external or cyclic references, `allOf`, conditionals,
or pattern-based objects use a JSON value editor. External references are not
fetched. **Edit configuration as JSON** can replace the whole form for more
complex inputs. The form guides entry; the server remains responsible for
constraints and checks between fields.

Click **Validate** to check the configuration, or **Export YAML** to check it and
show the resulting YAML with resolved defaults. The **Download config.yaml** link
saves that YAML. Both operations use `resolve_config`, report its key locations
and reasons on failure, and preserve fields that resolution leaves absent. Save
the YAML and pass it to `pixtreme-train train config.yaml --output runs/first`.
The builder does not start training. Its HTML, CSS, and JavaScript ship in the
wheel and load without an external CDN, font, or script.

Python callers with the extra installed can use
`pixtreme.train.common.webui.create_app(imports=["my_trainer"])`. Its HTTP endpoints
are:

| Endpoint | Input and response |
| --- | --- |
| `GET /api/components` | No input. Returns the same JSON document as `describe_components`, including startup `imports`, component schemas, and the common configuration schema. |
| `POST /api/validate` | The configuration object as a JSON request body. Returns `{"config": <resolved configuration>}`. |
| `POST /api/yaml` | The same JSON body. Returns the resolved configuration as `application/yaml`, with attachment filename `config.yaml`. |

Both POST endpoints insert the startup `imports` when omitted. If an `imports`
key is supplied, it must exactly match the startup list. HTTP requests never load
other modules. Configuration errors return status 422 and
`{"detail": "<resolver error, including its key location>"}`. JSON syntax errors
use FastAPI's normal 422 response. These operations import component dependencies
but do not initialize training, allocate GPU resources, or execute codecs.

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
configuration. `compression_artifacts` applies to the values it receives as they
are, without color conversion or clipping, so place a `color` step such as
`{colorspace: Rec.709, gamma: sRGB}` before it; otherwise the simulated codec
sees ACES2065-1/linear values, including values outside 0 to 1. A custom degradation operation supplies `validate(parameters,
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
ordered evaluation. Supply `paths` containing video files, saved `VideoIndex`
JSON files, or directories. Each directory contributes recursively collected
video files in path order, using `video_paths` below. This also applies to
`video_stream_evaluation`. These datasets produce TCHW float32 GPU clips. A model-specific
training/evaluation step must handle the temporal dimension and model state;
the default image step does not adapt a video model automatically.

`video_paths(directory, *, prefixes=())` returns a list of `Path` objects. It
matches `.mp4`, `.m4v`, `.mov`, `.mkv`, `.webm`, `.avi`, `.ts`, `.m2ts`, `.mts`,
and `.mxf` without regard to suffix case. It excludes files and directories
whose names begin with `.` beneath the requested directory, and does not follow
directory symlinks. Optional `prefixes` select filenames at every depth.
An empty selection raises an error naming the directory and extension conditions
before updates or evaluation. Collection only checks paths and suffixes; it
requires no GPU, codec initialization or `ffprobe`. A collected file must still
pass the existing indexing and decoder capability checks.

```python
from pixtreme.train.common import video_paths

sources = video_paths("videos")  # Includes videos nested below this directory.
selected = video_paths("videos", prefixes=("training_",))
```

Each directory expands at its position in `paths`. Directory contributions
exclude files explicitly listed anywhere in `paths` and files already collected
from another directory, comparing resolved paths. Explicit files keep their
order and repetitions, including when they follow a directory. Explicit video
and index JSON inputs retain their existing behavior and are not restricted by
the directory suffix or hidden-entry rules. The resolved configuration records
the expanded file list. Resolving the original directory configuration again
discovers added files; resuming after additions, removals or source changes uses
the same configuration and data-state correspondence checks as explicit files.

For whole-video evaluation, choose `video_stream_evaluation`. The model's
evaluation step opens each material with `with dataset.stream(material) as samples:`.
It receives every selected display frame once, including a short final delivery.
The source decoder, encoder and LQ decoder stay in one stream for that material;
changing `frames` only changes delivery boundaries. Crop coordinates and
compression settings are drawn once per material from its seed and material index.

| Evaluation data key or operation | `video_evaluation` / `VideoEvaluationDataset` | `video_stream_evaluation` / `VideoStreamEvaluationDataset` |
| --- | --- | --- |
| `frames` (default `1`) | Length of each independent clip; incomplete tails are dropped. | Maximum frames per delivery; the entire tail is returned, including materials shorter than `frames`. |
| `frame_step`, `time_range` | Select eligible complete clips. | Select display frames in the half-open time range, then take every `frame_step` frame starting at its first eligible frame. |
| `gop_aligned` | Every clip starts at a source keyframe. | Only the material's first selected frame is advanced to the first eligible source keyframe. |
| `crop_size`, `temporal_compression` | Draw a crop and codec settings per clip; each clip is an independent compressed stream. | Draw once per material; the crop, references, frame types and timing persist across deliveries. |
| `len(dataset)` | Number of complete clips. | Number of materials in `paths`. |
| Reading | `dataset[clip_index]`, including repeated random access. | `with dataset.stream(material_index) as samples:` followed by ordered iteration. Each context opens a fresh stream. |

Both evaluation types reject augmentation and scene mixing and support the RGB
and optional YCbCr LQ paths. Other keys keep the meanings in the video data table
below. The existing `video_evaluation` behavior and defaults are unchanged.

```python
from pixtreme.train.common import VideoStreamEvaluationDataset

dataset = VideoStreamEvaluationDataset(
    paths=["source.mov"], frames=5, seed=17, device="cuda",
    temporal_compression={"gop_length": 10, "frame_types": True},
)
try:
    for material in range(len(dataset)):
        with dataset.stream(material) as samples:
            for sample in samples:
                # TCHW CUDA tensors: sample["gt"] and sample["lq"].
                # Reset the model on LQ I frames and accumulate metrics per frame.
                prediction = evaluate_video_frames(sample)
finally:
    dataset.close()
```

`evaluate_video_frames` represents the caller's model evaluation. The model's
`TrainingStep.evaluate(..., context=...)` opens the stream contexts, including
when this data type is selected under `data.evaluation` in YAML. The runner
unwinds that method on a training signal. Use each context and its iterator on
the same thread. Exiting the context, even after `break` or an exception,
releases its source, encoder and LQ decoder without reading the remainder.
GPU retention covers delivery buffers and codec delay; the source is decoded
as the consumer advances. Only compressed bytes and timing/container metadata
pass through the CPU. Delivered tensors remain valid after context exit.
Callers retain those tensors for as long as they need them.

Stream samples carry the same fields, color, exact `pts`/`time_base`, source
`frame_numbers` and float `times` as video clips. `index` identifies the material
and `number` counts deliveries within it. `continuous` and source-based `reset`
continue across delivery boundaries; a boundary alone does not reset the model.
When compression requests `frame_types`, `frame_type` describes the continuous
LQ stream (B=`0`, P=`1`, I=`2`). The evaluation step selects model resets and
metric aggregation. An empty selected range is diagnosed before evaluation.
The existing source format, GPU and codec capability requirements also apply
to this data type.

To save the compressed LQ used by one evaluation stream, pass an MP4 destination
to the same context:

```python
with dataset.stream(material, lq_path="evaluation/lq.mp4") as samples:
    for sample in samples:
        prediction = evaluate_video_frames(sample)
```

`lq_path` defaults to `None`. Export requires `temporal_compression`; it copies
the same HEVC access units sent to the evaluation decoder, without another
encode or any change to samples or metrics. Codec selection and chosen settings
remain in `temporal_compression` and `sample["compression"]`. This also supports
B frames and the RGB and YCbCr LQ paths.

The MP4's presentation time for display frame `i` is `i / R`, starting at zero,
where `R` is the chosen `sample["compression"]["frame_rate"]` rational. The
source's original `pts`, `time_base` and `times` remain in the samples. A frame
rate override, source time range or stride can therefore make the MP4's times
differ from the source's absolute times. FFmpeg supplies decoding timestamps
from the access units' presentation positions and observed reordering depth.

Export stages compressed bytes and container metadata on disk beside the
destination; decoded pictures remain on the GPU. Allow disk space for the
temporary compressed input and MP4. The destination's parent directories are
created when needed. After the iterator is exhausted and the context exits
successfully, ffmpeg performs stream copy and the completed MP4 atomically
replaces the destination. A `break`, unconsumed iterator or evaluation exception
discards the temporary export and preserves any previous destination. A mux or
file error also preserves the previous destination and raises an exception.
Use a separate destination for each material whose LQ you want to retain.

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
| `lq_resize` | `rgb`; `ycbcr` selects source YCbCr reduction before encoder conversion and requires `temporal_compression`. Applies to training and evaluation. |
| `decode_retries` | `2`; nonnegative integer number of extra attempts after a source or LQ decoding failure, excluding the first attempt. `0` disables retries. Applies to all video data types. |
| `index_paths` | Optional mapping from expanded video file paths to index JSON destinations. Keys name individual videos, including directory-collected videos, using the spelling returned by `video_paths`; directory and saved-index JSON keys are rejected. Omit it to keep rebuilding video paths without saving indexes. Available for training and evaluation. |
| `patch_size` | Training only: `null` (return the entire prepared clip), or final GT patch size divisible by `scale`. Patches are cut after temporal compression. When set, it must fit `crop_size` and every source. |
| `patches_per_clip` | Training only: `1`; positive integer number of consecutive samples from each decoded/compressed clip. Values above `1` require `patch_size`. Independent of `train.batch_size`; a clip's samples may span batches. |

Each decode retry discards incomplete output and uses a fresh decoder with the
same material, time interval and encoded input. Warnings identify the material,
time interval, retry number and cause; exhaustion returns the material and cause
to the caller. Sample selection, crops, compression choices and augmentation do
not change. Configuration, source I/O and encoding failures are not retried.
Whole-video evaluation keeps samples already delivered, discards unfinished
deliveries and skips the delivered prefix when decoding again. Compressed LQ
access units are staged in a temporary file and replayed without encoding again;
the file is closed on completion, failure or early exit. A decode call that never
returns is handled by the training supply time limit, separately from retries.

To reuse indexes across starts, choose a destination for each video. The names
and directories belong to the caller; the foundation does not impose a cache
filename. Destination directories must already exist. For a source named
`training.mov`, set `index_paths: {training.mov: .training.mov.index.json}` in
its data entry, as in the training example below. Sources omitted from the
mapping are scanned without saving an index.

For `paths: [videos]`, a collected source `videos/training.mov` can use
`index_paths: {videos/training.mov: indexes/training.json}`. A directory itself
cannot be a mapping key. Use the public `video_paths("videos")` to enumerate
sources when choosing destinations programmatically. Existing unknown-key and
saved-index JSON validation also applies to directory inputs.

Only new videos and videos with missing, unreadable, unsupported or mismatched
indexes are scanned with `ffprobe`. Correspondence uses the resolved source
path, byte size, and modification time floored to whole seconds. Differences
within the same second also remain accepted by subsequent reads and resume.
Changing the path, size, whole-second modification time or indexed frame timing
still prevents resuming with different material. Removed videos' caches are
unused. Indexes are fully written and synced to a sibling temporary file before
replacement. Save failures warn with the source, destination and reason, then
continue with the freshly built in-memory index. An index destination must
differ from its source video.

Direct callers use the same behavior through
`VideoIndex.build(source, cache_path=destination)`. Without `cache_path`, `build`
always scans and does not save. `VideoIndex.save(destination)` also uses complete
file replacement, but reports save errors to its caller. `VideoIndex.load(path)`
loads a saved index without scanning and reports errors rather than rebuilding.
Existing version 2 JSON indexes remain readable. Version 1 indexes lack
container color metadata and require rebuilding: pass the original video and
destination to `build`, or use video paths with `index_paths`. Passing an old
JSON file directly in `paths` reports its unsupported format.

Video data state remains version 1. New fingerprints normalize the source
modification time to whole seconds and retain the full indexed metadata and
timing. Older exact-nanosecond fingerprints are accepted when their original
version 2 index reproduces the saved fingerprint. Keep that index when resuming
an older checkpoint; if it has been rebuilt with different fractional seconds
and no longer matches, start a new run. No index or checkpoint is silently
reinterpreted as a different material.

`crop_size` keeps its existing meaning: it selects the GT region before LQ
reduction and temporal compression. `patch_size` selects a smaller region after
those operations, then applies each sample's augmentation to every field and
frame. A patch's coordinates and size must align with every field's pixel grid.
With `patch_size: null` and `patches_per_clip: 1`, pixels, sample records, and
random choices are unchanged.

To reduce source YCbCr without passing the LQ pixels through RGB, set
`lq_resize: ycbcr` in the video data entry and enable `temporal_compression`.
The reader preserves the source PixelBuffer's format, bit depth and color
declarations. On the GPU, pixtreme's `from_pixel_buffer` expands it to a float32
YCbCr Frame, `transform.resize` applies area reduction, and `to_pixel_buffer`
converts once to the encoder input. Source primaries, transfer, matrix, range
and chroma position govern unpacking; the selected output color, matrix, range,
bit depth and left chroma position govern the encoder input and its HEVC
declaration. Chroma is interpolated while expanding subsampled source pixels to
a Frame. This route does not retain the source plane grids during reduction.
GT and decoded LQ remain RGB in the configured `colorspace`/`gamma`. Crops still
precede reduction; patches and augmentation still follow compression. The
default `lq_resize: rgb` retains the existing RGB conversion and reduction.
This option requires pixtreme's public `from_pixel_buffer`, `transform.resize`
and `to_pixel_buffer` interfaces; it is supported with pixtreme 2.27.5.

Supported version-two RGB checkpoints written before `lq_resize` was added
retain their implicit `rgb` meaning, for both training and evaluation data.
They can resume with the option omitted or explicitly set to `rgb`. Resume
compares data options through the registered data type's configuration resolver,
including the existing single-patch defaults. Changing between `rgb` and
`ycbcr` changes preprocessing and is rejected on resume; two otherwise matching
`ycbcr` configurations remain compatible. Checkpoint files and saved training
states are preserved during this comparison.

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
    index_paths: {training.mov: .training.mov.index.json}
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
| `frame_rate` | `null`: source average display rate divided by `frame_step`; a positive rational such as `"30000/1001"` overrides it. For one-frame sources, the rate is the reciprocal of the frame's packet duration in seconds, divided by `frame_step`. A missing or nonpositive duration requires an explicit rate. |
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

Direct Python operations include `VideoIndex.build(source, *, cache_path=None)`,
`VideoIndex.save/load`, `VideoFrame`,
`ClipSampler`/`ClipSelection`, `VideoField`/`VideoSample`,
`VideoClipReader.read/close`, and
`VideoDataset`/`VideoEvaluationDataset`/`VideoStreamEvaluationDataset`.
`VideoClipReader.read_pictures(selection)` returns source PixelBuffers in the
selection's display order, preserving their format, declarations and code bits.
The selection retains exact PTS/time_base correspondence. Each picture owns a
GPU copy, so it remains usable after another read or `close()`; callers release
the returned pictures when finished. The method uses the same bounded source
decoder sessions as `read`.
`TemporalCompression.resolve_config` validates settings without running codecs;
`select(generator)` returns scalar settings, `validate(...)` checks capabilities,
and `apply(lq, generator, *, frame_rate, ...)` returns fields and chosen settings.

`TemporalCompression.close()` releases every LQ decoder session retained by
`apply`, including sessions opened by other sample threads. It is safe before
the first `apply` and on repeated calls; a later `apply` opens a new session as
needed. A custom data type that owns this component forwards cleanup from its
own `close()` method:

```python
from pixtreme.train.common import TemporalCompression

class CustomVideoData:
    def __init__(self):
        self.temporal_compression = TemporalCompression({"gop_length": 10})

    def close(self):
        self.temporal_compression.close()
```

Call the custom data type's `close()` when sample work finishes, including on
errors. `VideoDataset.close()` already closes its temporal compression component.

`encode_hevc(lq, parameters, *, frame_rate, colorspace="Rec.709", gamma="Rec.709")`
encodes a nonempty TCHW floating CUDA RGB tensor to Annex B HEVC bytes. Pass
settings from `TemporalCompression(...).select(generator)`.
`encode_hevc_pictures(pictures, parameters, *, frame_rate, scale=1, crops=None,
colorspace="Rec.709", gamma="Rec.709")` accepts a nonempty sequence of source
PixelBuffers on one GPU. `scale` is a positive integer divisor of the selected
dimensions. Optional `crops` has one `(top, left, height, width)` or `None` per
picture, in source luma coordinates; resulting dimensions must match. The call
expands YCbCr, crops and reduces it on the GPU, then uses the same NVENC
submission/drain/close loop as `encode_hevc`. `TemporalCompression.apply` also
accepts these pictures with `scale` and `crops`, returning decoded RGB fields
and selected settings through the same LQ decoder sessions. For example:

```python
pictures = reader.read_pictures(selection)
fields, chosen = compression.apply(
    pictures, generator, frame_rate=frame_rate, scale=2,
    colorspace="Rec.709", gamma="Rec.709",
)
```

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

Use `frame_from_tensor(values, color)` from `pixtreme.train.common` to pass a
network output to the common metrics. `values` must be a CUDA `torch.float32`
RGB tensor with shape `(3, height, width)` and positive dimensions. Contiguous
tensors and noncontiguous CHW views are accepted; split batches and video clips
into individual images before calling. `color` must specify `colorspace`
(primaries) and `gamma` (transfer) using pixtreme's supported color names. The
function copies the values on the same GPU without changing color, normalizing,
or clipping, including values outside `[0, 1]`. CPU tensors, other dtypes,
incompatible shapes and missing or unsupported color names raise `ValueError`.
Network outputs with `requires_grad=True` are accepted. The evaluation copy
does not track gradients; the original tensor's autograd state is preserved.

```python
from pixtreme.train.common import frame_from_tensor, measure

color = {"colorspace": "Rec.709", "gamma": "sRGB"}
# reference and prediction are CHW float32 CUDA tensors produced on the current stream.
reference_frame = frame_from_tensor(reference, color)
prediction_frame = frame_from_tensor(prediction, color)
metrics = measure(0, reference_frame, prediction_frame)
```

The copy uses the current torch stream for the tensor's device and keeps
temporary tensor storage alive until the copy completes, including storage
borrowed through DLPack. The function waits for the copy to finish before
returning the Frame. When another stream
produces the tensor, order it with `torch.cuda.current_stream(values.device).wait_stream(producer)`
before calling. The returned Frame owns its pixels independently of later
source writes ordered after the copy. The default evaluation step uses the
same operation. Noise degradation and color clipping with
`value_clipping="color_and_metrics"` also wait for their tensor-to-Frame copies.
Callers using the private `_frames._from_tensor` should import
`frame_from_tensor` from `pixtreme.train.common`; the private name remains an
alias to the same implementation. `rgb_frame` continues to accept Frames.

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
The registered step's `check_checkpoint` validates model records first; rejected
records never become an inference network or a published weight artifact.

`export_weights(checkpoint, output, *, name="main", ema=True, metadata,
transform=None, inference_metadata=None)` writes a new artifact directory containing `weights.safetensors`
and `manifest.json`, with source checkpoint SHA256, selected network/weights and
completed update count. The manifest's `metadata` must declare positive integer `scale`,
`input_color` and `output_color` (`colorspace` and `gamma`), `normalization`, and
`value_clipping`. Optional `transform(state_dict)` converts tensor names/values
before tensoracle writes safetensors. The artifact contains inference weights,
not optimizer, RNG or resumable training state.
Export and calibration deserialize the source checkpoint once and share that
validated content for weight selection and provenance. The SHA256 is computed
from the same open source file before deserialization.

Pass `inference_metadata` to put the trainer's string keys and string values
directly into `weights.safetensors`'s `__metadata__` header. An inference consumer
can then read these conventions from that file alone. This mapping is the source
for the header and is not copied into `manifest.json`; the existing structured
`metadata` argument continues to describe the manifest. The trainer chooses both
declarations and keeps any shared conventions consistent. Export does not infer
header fields from the checkpoint, convert values to strings, or compare header
fields with the manifest or saved model records.

For example, a trainer with cda-vsr-style inference conditions can declare its
header's wire values explicitly:

```python
from pixtreme.train.common import export_weights
from safetensors import safe_open

export_weights(
    checkpoint,
    artifact_directory,
    metadata={
        "scale": 2,
        "input_color": {"colorspace": "Rec.709", "gamma": "sRGB"},
        "output_color": {"colorspace": "Rec.709", "gamma": "sRGB"},
        "normalization": "float32 scene values",
        "value_clipping": "codec_only",
    },
    inference_metadata={"residual_unit": "normalized_rgb", "scale": "2", "gate": "true"},
)
with safe_open(artifact_directory / "weights.safetensors", framework="pt") as weights:
    inference_conditions = weights.metadata()
```

The trainer's step can save `residual_unit`, `scale`, and `gate` in its model
record and validate that record in `check_checkpoint`. Export runs that check
before applying weights or calling `transform`; an incompatible record prevents
publication even when `inference_metadata` is valid. The trainer remains
responsible for choosing header values that describe the checked model.

Omitting `inference_metadata` or passing `None` preserves metadata-free weight
bytes; `{}` writes an empty header mapping. Tensoracle validates this argument:
non-string keys or values raise `TypeError`, and invalid UTF-8 or an oversized
header raises `ValueError`. Empty strings, Unicode, whitespace and embedded NUL
characters are preserved. Export propagates the converter's exception with
`inference_metadata` key context in its exception notes and publishes no partial
artifact.

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
| `video_paths(directory, *, prefixes=())` | Recursively lists supported visible video files in path order, rejecting an empty selection. Prefixes select filenames at every depth. Directory inputs to video datasets use this collector; explicit files retain their order and repetitions. |
| `warmup_cosine_factor(completed, *, warmup, total, minimum)` | Returns the learning rate coefficient for `completed` updates. Set `warmup=0` for cosine decay from `1.0` at zero updates to `minimum` at `total`; positive warmup lengths first increase the coefficient linearly. Requires `0 <= warmup < total` and `0 <= minimum <= 1`. |
| `update_ema(average, model, *, decay)` | Averages floating state and copies nonfloating state. `check_finite_loss(loss, *, iteration)` and `check_finite_gradients(parameters)` reject nonfinite values; finite gradients are not clipped. |
| `release_gpu_memory(device)` | Returns unused torch and CuPy allocator blocks for a CUDA device. |
| `training_signals()` | Context manager exposing the first interruption to the training loop and escalating a second signal. `worker_signals(worker_id)` configures a DataLoader worker; `stop_loader_workers(iterator)` requests shutdown, waits up to five seconds and reaps survivors. |

```console
pixtreme-train train config.yaml --output runs/continued --resume runs/first/checkpoint_00000500.pt
```

Resume permits changes to `train.iterations` (must exceed completed updates),
`train.device`, `train.sample_threads`, `train.supply_timeout`, `train.channels_last`,
`train.cudnn_benchmark`, `train.compile`, and `notes`. Other settings must match the
saved configuration, including data sources, degradation, color, batch size and
prefetch depth. Version 2 checkpoints lacking `supply_timeout` use `600` seconds;
those lacking `sample_threads` use `1`; missing
network execution options mean `channels_last: false`, `cudnn_benchmark: false`,
and `compile: null`. These options may change on resume in either direction
without converting checkpoint weights. Model
steps may further restrict resumption. The next sample follows the last completed
update; unused prefetched samples are regenerated.

An omitted `train.seed` on resume uses the seed saved in the checkpoint and does
not choose a new one. An explicitly supplied seed must match the saved value.

The same sources, configuration, environment and seed reproduce material/sample
selection and preprocessing random choices. Bitwise identity of model updates
across GPUs, library versions or arithmetic modes is not promised. Codecs can
also depend on hardware/driver behavior; compare numerical outputs under matched
conditions and a tolerance suitable for the operation.

The first SIGINT/SIGTERM stops new batch production, finishes an in-progress
update, and saves the last completed state. No checkpoint is written before the
first completed update or when no update has advanced since the last checkpoint.

An interrupted evaluation does not become a successful report. A second signal
exits without waiting for saving. The first signal also consolidates
`training.jsonl` in the same format as normal completion, containing only
completed updates and completed evaluations. Before the first completed update,
the file is empty. Incomplete updates are excluded, and an interrupted
evaluation contributes no evaluation values or report. A second signal skips
this consolidation when it has not started and can interrupt it in progress.

If training batch supply or evaluation data supply does not return within
`train.supply_timeout` seconds, the command stops learning and exits with code
`75` (temporary failure; restart may succeed). The wait starts when the learner
requests the next batch, even when prefetch is disabled. Evaluation bounds each
indexed read, iteration and entry/exit/iteration of a dataset's `stream()` context;
other custom dataset methods and evaluation calculations are outside this limit.
Evaluation calculations and hooks stay on the learning thread; data acquisition
inherits the caller's CUDA stream and gradient/autocast settings.

The command writes the normal `training.jsonl` containing completed updates,
discards partial evaluation metrics, and saves the last completed update unless
it was already saved. Before update one, it writes no checkpoint. Saving after
a supply timeout has the same time limit; if saving also stalls, the command
exits without publishing an incomplete checkpoint. It reports the dataset,
reason, saved checkpoint (or that none was saved), and how to continue on stderr.
The command flushes its output and completed records and exits without waiting
for unfinished native calls. Continue with a new output directory, for example:

```console
pixtreme-train train config.yaml --output runs/retry --resume runs/first/checkpoint_00000500.pt
```

If no checkpoint was saved, retry from an earlier checkpoint, or restart the run.

Python callers of `train()` receive `SupplyTimeoutError`, a `RuntimeError`
exported from `pixtreme.train.common`. Its `target` names the batch or evaluation
dataset, `reason` describes the expired wait, and `checkpoint` is a `Path` to the
saved or reused checkpoint, or `None` when none is available. After catching this
exception, end the process and start a new process using `--resume`; do not
continue training in that process, because the CUDA state may be unusable.
Unfinished supply threads do not prevent the caller's process from exiting.

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
          - color: {colorspace: Rec.709, gamma: sRGB}
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
