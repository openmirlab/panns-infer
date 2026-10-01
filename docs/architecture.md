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
