# Inference extraction gate

Before the session facade was implemented, the extracted 89-line model passed
all five original CPU fixtures bit-exactly under Torch 1.13.1. Torch 2.7.1 CPU
and CUDA pass every score, embedding and logmel component at rtol=atol=1e-4;
top-ten class indices are identical. No stochastic inference operator remains.

A read-only deep-module audit covered all eight principles before facade work:

1. Checkpoint geometry and operators are owned by the private model; callers
   receive only native scores and embeddings.
2. The package facade is lazy; private operators stay in `_impl`.
3. The model receives waveforms and a checkpoint path explicitly.
4. One cohesive 89-line CNN14 module avoids both a god file and tiny operator files.
5. Normal `nn.Module`, strict state dictionaries and inference mode fit Torch.
6. Every original eval operator and parameter name is retained; train-only
   augmentation, mixup and dropout branches are absent.
7. Historical librosa resampling is explicitly excluded from the parity claim.
8. Each module can be described directly from its file header.

No errors or unresolved warnings were found in this bounded extraction audit.
Runtime scans found no training/evaluation/dataset dependency or executable
surface. Research baseline tooling imports the unchanged upstream only when
explicitly run, and is excluded from the wheel.

Torch 1.13.1 → 2.7.1 changes backend floating-point reductions slightly. CPU
logmel is exact, CUDA logmel differs by at most 1.53e-5, scores by 4.77e-7 and
embeddings by 6.03e-6 on the recorded fixture set. These are controlled
cross-version differences, not a changed frontend algorithm; checkpoint filters
and eval graph remain identical. These measurements are not bit-exact claims
on other builds/hardware.

# Completed package verification

The public facade and CLI were tested from a wheel built **from sdist**:
Python3.10/Torch2.7.1+cpu and Python3.11/Torch2.7.1+cu126 with NumPy2.2.6.
All five file inputs match original native527scores and2048embedding components;
file/array/tensor results agree, fullCLIJSON is checked against API output,
network access is forbidden for offline resolution, and release/reload plus
terminal close pass. CPU and CUDA each pass52tests. Installed result vectors
remain available and JSON-serializable after releasing the resident model.

Archive inspection confirms registry plus MIT/NOTICE/torchlibrosa license
inclusion and absence of weights, audiofixtures, research tools or tests from
the wheel. The sdist includes authorizedCC0 fixtures and test/tool sources,
without modelweights, environments, private recordings or local reports.
These are package E2E checks, not a hosted-service or resource-budget acceptance.

# Additive lifecycle and normalization changes

A scoped boundary review found one provenance warning: expected mutable cache
metadata included local paths. `loaded_checkpoint_info()` resolves this with
a path-free successful-load snapshot, shared session lock and defensive copy;
it never performs file inspection or model loading.

Optional short-input padding is owned by the existing normalization boundary,
after unchanged channel averaging and resampling. Default rejection remains;
source coverage and synthetic padding are recorded separately. CPU/CUDA tests
compare every score and embedding bit-exactly against independently normalized
and manually right-padded inputs at 16, 32 and 44.1 kHz for all input types.

The three-function TorchAudio 2.7.1 resampling closure is copied verbatim, with
its pinned BSD-2-Clause source/license retained. Seeded CC0 CPU/CUDA fixtures
were recorded before extraction; source equality and live float32/float64 A/B
checks pass bit-exactly. The runtime has no TorchAudio dependency. The narrower
pure-Torch boundary introduces no algorithm, precision or ordering changes.
Python 3.13/Torch 2.13 CPU, without TorchAudio installed, passes 72 tests;
12 recording-environment/reference-only tests skip explicitly at that checkpoint.
Container resource budgets remain unverified.

# Target CUDA runtime and duration feasibility

A local CUDA 13.0/Ubuntu 24.04 Docker container on RTX 4090 (driver 595.84)
verified the installed wheel with Python 3.13.15, Torch 2.13.0+cu130 and NumPy
2.5.3. TorchAudio imports were actively blocked. The full real-checkpoint GPU
suite passed 74 tests; 12 optional original-reference or recording-build checks
skipped explicitly. Installed API/CLI E2E passed all five original fixtures,
all 527 scores and 2,048 embedding components, opt-in padding against manual
normalized input, resident provenance and release/reload/terminal close.

Synthetic Gaussian inputs of 1, 10, 30, 60 and 120 seconds completed with
finite scores. Selected Torch allocator measurements were:

| Input duration | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|
| 10 seconds | 758,325,248 | 1,184,890,880 |
| 120 seconds | 944,642,560 | 1,184,890,880 |

After releasing the model, its weak reference cleared and allocated memory
returned to 8,519,680 bytes, matching an independently established framework
baseline from a plain 32×32 matrix multiplication. The container process
terminated after verification. The measured image content ID was
`167ea96ddbbd7f96dab67217546e22fc810fc71080fd412506673193e3d784eb`.

These are native execution-feasibility and Torch allocator observations from
single forwards with mixed warmup state. They are not whole-GPU/NVML peaks,
host RAM, deployment budgets, recognition quality or latency benchmarks.
