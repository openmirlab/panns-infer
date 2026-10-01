# panns-infer

PANNs CNN14 AudioSet tagging and native audio embeddings, as an independent inference-only package.

## Why this exists

[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) introduced reusable
pretrained audio neural networks. Its research repository includes training,
evaluation and historically pinned dependencies. This package extracts CNN14
inference for modern Python and Torch while preserving its checkpoint and
native outputs.

## Acknowledgments

**All model credit belongs to the original PANNs authors. Please support and cite their work.**

- [Qiuqiang Kong and the original PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn).
- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D. Plumbley,
  verified against the [original paper](https://arxiv.org/abs/1912.10211).
- [Zenodo record 3987831](https://zenodo.org/records/3987831), the original pretrained checkpoint host.
- [AudioSet](https://research.google.com/audioset/), its 527-class ontology and label mapping.

## Citation

```bibtex
@misc{kong2020pannslargescalepretrainedaudio,
      title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition}, 
      author={Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
      year={2020},
      eprint={1912.10211},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/1912.10211}, 
}
```

## Features

- Clip-level AudioSet tagging: all 527 native sigmoid scores, display names and class MIDs.
- The complete native 2,048-dimensional float32 clip embedding; no implicit L2 normalization.
- CPU and CUDA, with explicit devices and independent reusable sessions.
- Verified official weights, custom local paths, configurable artifact registry and offline cache.
- Full JSON CLI output including the embedding, input metadata and checkpoint provenance.

## Scope

This first version supports only `Cnn14_mAP=0.431.pth` from the original
32 kHz CNN14 model. Scores are independent, uncalibrated class scores: they
need not sum to one and are not calibrated confidence estimates. A clip may
contain multiple sound classes. An embedding represents the whole clip,
not a sequence of timestamped events.

Other PANNs architectures, 8/16 kHz checkpoints, framewise sound event detection,
captioning and quality claims on a new dataset are not supported. Training,
fine-tuning, evaluation frameworks and dataset downloading are out of scope,
forever. This repository contains inference code and verification fixtures.

## Install

Python 3.10 or newer; Torch and torchaudio 2.7.1 or newer; NumPy 2 is supported.
This package is currently private and has not been published to PyPI. From an
authorized checkout:

```bash
python -m pip install .
```

Select matching Torch/torchaudio builds for your CPU or CUDA environment using
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
Torch 2.7.1 CPU/CUDA model parity is tested locally; the CI Python matrix is
3.10–3.13. Versions above the tested floor are not capped, but each new backend
combination still needs the numerical verification below.

## Quick Start

```python
from panns_infer import PANNsSession

with PANNsSession(device="auto") as session:
    session.load()  # downloads missing assets, verifies SHA256, then becomes ready
    result = session.infer("audio.wav")
    for tag in result.ranked_tags[:10]:
        print(tag.label, tag.score)
    print(result.embedding_shape)  # (2048,)
    print(len(result.embedding))  # 2048 native components
```

Importing the package, constructing a session and entering its context do not
load weights or Torch. `load()` is explicit; repeated `infer()` calls reuse
the same resident model. The result owns ordinary Python values and remains
valid after the session closes.

## Input contract

`infer(audio, *, sample_rate=None)` accepts a path, NumPy array or Torch tensor.
Files supply their own metadata. Bare waveforms require a positive integer
`sample_rate` and floating point samples; shapes are `(samples,)` or
channel-first `(channels, samples)`. Integer PCM arrays, empty input, NaN,
infinity and batched 3D input are rejected. A supplied rate that disagrees with
file metadata is rejected.

```python
import numpy as np
from panns_infer import PANNsSession

waveform = np.zeros(32000, dtype=np.float32)
with PANNsSession(device="cpu") as session:
    session.load()
    result = session.infer(waveform, sample_rate=32000)
```

SoundFile decodes to float32; channels are averaged on CPU, then torchaudio's
functional resampler converts to 32 kHz before transfer to the selected model
device. File, array and tensor inputs share this boundary. No clipping, gain
normalization or padding is added. CNN14 requires at least 9,920 normalized
samples (about 0.31 seconds) to survive its five temporal pooling stages.
There is no claimed maximum duration; cost grows with input duration and
long recordings have not been certified for deployment or quality.

This normalization policy is explicit. Historical upstream `librosa.load`
resampling is not claimed to be identical. Numerical model parity is checked
on the same recorded normalized waveforms; normalization is checked separately.

## Results and embeddings

`TaggingResult` is a frozen dataclass:

| Field | Meaning |
|---|---|
| `scores`, `labels`, `label_ids` | 527 values in original AudioSet index order, names and MIDs |
| `ranked_tags` | Read-only descending ranking; equal scores use class index |
| `embedding` | All 2,048 native float32 components as a tuple of Python floats |
| `embedding_shape`, `embedding_dtype`, `embedding_meaning` | `(2048,)`, `float32`, native representation description |
| `audio` | Original rate, channel count, samples, duration and model sample rate |
| `checkpoint` | Repository, release identity, upstream revision and artifact hashes |

The embedding is the original CNN14 `embedding` output: max-over-time plus
mean-over-time features, followed by `fc1` and ReLU. Upstream evaluation dropout
is the identity. It is not L2 normalized and is not a timestamped sequence.
Use any task-specific normalization explicitly after obtaining this vector.
`dataclasses.asdict(result)` can be serialized directly with `json.dumps`.

## Session lifecycle and devices

| Operation | Behavior |
|---|---|
| `load()` | Idempotent while ready; successful loading alone makes it ready |
| `infer()` | Ready-only; no hidden cold start; operations serialize per session |
| `release()` | Drops the resident model and permits a subsequent `load()` |
| `close()` | Terminal and idempotent; context exit calls it |
| `status` | `new`, `loading`, `ready`, `busy`, `failed`, `released`, `closed` |
| `device` | Concrete loaded device, or `None` when no model is resident |
| `cache_info()` | Read-only asset paths/existence/expected identity; never downloads |

`device` accepts `auto`, `cpu`, `cuda` and `cuda:N`. Auto selects CUDA device 0
when available, otherwise CPU. Explicit unavailable CUDA choices raise;
there is no silent fallback. Sessions do not change global Torch precision
settings. CPU thread counts belong to the caller. `release()` removes model
references and requests unused CUDA allocator blocks be released; Torch's
context/workspaces can remain resident. Reclaiming an entire worker process
is a separate deployment concern.

## Weights, integrity and offline use

The packaged `config/checkpoints.json` is the runtime source of truth. Default
weights are the official [Zenodo 3987831](https://zenodo.org/records/3987831)
CNN14 artifact (327,428,481 bytes):

```text
SHA256 0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31
Official MD5 541141fa2ee191a88f24a3219fff024e
Source revision d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
```

AudioSet labels are obtained from the official Google-hosted CSV and verified
as SHA256 `cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429`.
Load verifies every file; strict state loading uses `torch.load(weights_only=True)`.
Missing managed assets download atomically under a file lock. Existing corrupt
assets raise rather than being silently replaced. Missing explicit weights
raise even in online mode. Cache inspection reports `integrity="not_checked"`:
existence alone is not proof that a future load will validate.

Default cache root is `~/.cache/panns-infer`, overridable through
`PANNS_INFER_CACHE` or `cache_dir`. The revision subdirectory is
`cnn14-zenodo-3987831`. Populate it once online, then use:

```python
from panns_infer import PANNsSession

with PANNsSession(cache_dir="/path/to/cache", offline=True, device="cpu") as session:
    session.load()
    result = session.infer("audio.wav")
```

Alternatively set `checkpoint_dir` to a read-only directory containing
`Cnn14.pth` and `class_labels_indices.csv`. If labels are missing there, the
managed label cache is used; offline mode requires it already exists.
`checkpoint_config` accepts a custom JSON registry with the same validated
schema to override URLs, expected hashes/sizes and provenance generically.
The model geometry remains CNN14; arbitrary other model checkpoints are not supported.

## CLI

The command name matches the distribution name:

```bash
panns-infer audio.wav --device cpu --output tagging.json
panns-infer audio.wav --device cuda:0 --cache-dir /path/to/cache --offline
panns-infer --help
```

Default output is JSON on stdout; `--output` writes the same complete payload.
All 527 scores/labels/MIDs and all 2,048 embedding components are included,
with metadata and provenance. Output is never silently truncated to top-k.
`--checkpoint-dir`, `--checkpoint-config` and `--cache-dir` match the Python
constructor. Invalid input or loading errors exit with status 2.

## What this project will NEVER bundle

No model weights, original/private audio recordings, datasets, training or
evaluation runtime, notebooks, upstream source tree, local environments,
cache directories, secrets or machine-specific probe reports. Generated CC0
verification fixtures and their original-model activations may be committed;
they are excluded from the wheel. Original source is a read-only reference.

## Development

```bash
python -m pip install '.[dev]'
python -m pytest -q
PANNS_TEST_CHECKPOINT_DIR=/path/to/verified/assets PANNS_TEST_DEVICE=cpu python -m pytest -q
PANNS_TEST_CHECKPOINT_DIR=/path/to/verified/assets PANNS_TEST_DEVICE=cuda:0 python -m pytest -q
PYTHONPATH=src python tools/verify_parity.py --checkpoint /path/to/Cnn14.pth --device cpu
python -m ruff check .
python -m ruff format --check .
python -m build
```

Real-model tests are explicitly skipped without a local checkpoint directory;
o test silently downloads weights. CI smoke tests cover Python 3.10–3.13.
An opt-in model workflow downloads checksum-verified official weights and runs
CPU integration tests. Golden CPU/CUDA fixtures were recorded under Torch
1.13.1 before porting; every native score and embedding component is compared
at explicit cross-backend tolerances, with exact top-ten class indices.
See [baseline details](docs/baseline.md) and [extraction evidence](docs/architecture.md).

Build wheel from sdist and test the installed wheel's public API and CLI in
an isolated environment, outside the source import path:

```bash
python -m pip install dist/panns_infer-0.1.0-py3-none-any.whl
python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/verified/assets --device cpu
```

This exercises files, tensors and arrays, full JSON outputs, offline resolution,
reuse, release/reload and terminal close. Synthetic fixtures prove regression
behavior, not real-world accuracy or a container resource budget.

## License

Code is MIT; original copyright notices are preserved in [LICENSE](LICENSE)
and [NOTICE](NOTICE). Original CNN14 checkpoint weights are **CC BY 4.0** as
stated by the [Zenodo record](https://zenodo.org/records/3987831), distinct from
the code license. AudioSet dataset and ontology terms are separate (CC BY 4.0
and CC BY-SA 4.0 respectively); see [AudioSet downloads](https://research.google.com/audioset/download.html).
Weights are downloaded from their original hosts and are not redistributed here.

## Support

For package bugs, report the Python/Torch/torchaudio build, selected device,
checkpoint provenance and a reproducible permitted input. For the research
model and paper, refer to the original PANNs repository above.
