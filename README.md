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

## License

MIT. See [LICENSE](LICENSE).
