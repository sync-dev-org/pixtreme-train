# pixtreme.train.common

The pure Python distribution `pixtreme-train` provides the import package
`pixtreme.train.common`: training components that do not depend on a particular
model, shared by the pixtreme trainer modules.

The wheel owns only the `pixtreme/train/common/` subtree. The namespace root
`pixtreme/__init__.py` is owned by [pixtreme](https://github.com/sync-dev-org/pixtreme),
and each trainer module owns its own `pixtreme/train/<module>/` subtree.

## Installation

Use Linux x86_64 with CPython 3.12 or later. Install the distribution from PyPI:

```console
pip install pixtreme-train
```

For a project managed with uv:

```console
uv add pixtreme-train
```

PSNR and SSIM measurement needs an NVIDIA GPU with CUDA 13, as required by
pixtreme. The other components also run without a GPU. The `prores_lt`, `h264`
and `hevc` encode steps, and reading 16-bit RGB images, need an `ffmpeg` command
on `PATH` with the `prores_ks`, `libx264` and `libx265` encoders.

## Training from a configuration

A trainer module registers its networks, losses and training steps by name, and
a YAML configuration combines them:

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

```yaml
imports: [my_trainer]
networks:
  main: {type: tiny}
losses:
  pixel: {type: l1}
optimizers:
  main: {type: adamw, networks: [main], lr: 2.0e-4}
data:
  train: {type: image_degradation, directory: /data/train, region_size: 64, patch_size: 64, batch_size: 8}
degradation: {name: identity, seed: 0, steps: []}
train: {iterations: 1000, seed: 0, device: cuda, checkpoint_interval: 500}
```

```console
pixtreme-train train config.yaml --output runs/first
```

The command writes the resolved configuration, one JSON line per update and the
checkpoints under the output directory, and prints the path of the last
checkpoint. `pixtreme.train.common.load_network` rebuilds a trained network from
a checkpoint for inference or export.

## License

MIT. See [LICENSE](LICENSE).
