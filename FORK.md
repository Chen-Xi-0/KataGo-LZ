# KataGo-LZ (Leela Zero fork)

This is a personal fork of [KataGo](https://github.com/lightvector/KataGo) that can load **Leela Zero (LZ)
weights directly** — no retraining and no offline Python conversion — plus a few experimental additions for
LZ-style play.

- Upstream base commit: `d91ea855`
- `main` = this fork (26 commits on top of upstream); `master` = upstream mirror
- License: GPLv3 (see `LICENSE`)

## What this fork adds

- **In-memory LZ text-weight converter** (`cpp/neuralnet/lzconverter.cpp`): sniffs LZ v1/v2 text weights
  and builds an mv18 `ModelDesc` directly.
- **Auxiliary score/ownership evaluator** (`--aux-model` / `auxModelFile`): a second KataGo model that
  only provides score/ownership for reporting; it does not feed back into search.
- **LZ dynamic komi** support (virtual komi derived from playout doubling advantage).
- **Ladder (征子) correction** aligned with the reference LZ fork (`LK/lzl-src`) semantics:
  capture boost, defence/offence rules with solver depth thresholds, root-armed value blending, and
  `lzLadder*` config keys. Defaults target long ladders only.
- **Experimental `auxValueWeight`** (`0..1`, default `0`): blends the auxiliary model's value into the
  search value while keeping the LZ policy. Off by default.

## Build (Windows / MSVC)

Toolchain used: MSVC 14.38, Windows SDK 10.0.28000, CMake + Ninja.

- OpenCL (recommended for LZ on this machine):
  `-DUSE_BACKEND=OPENCL` with zlib linked; requires the GPU driver's `OpenCL.dll`.
- CUDA (experimental): `-DUSE_BACKEND=CUDA`, CUDA 12.6 + cuDNN 9.
- Eigen CPU: `-DUSE_BACKEND=EIGEN` (slow; useful for reference/testing).

The MSVC builds need the Visual C++ 2015-2022 x64 runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`,
`VCRUNTIME140_1.dll`).

## Notes

- **Weight files are not included** (LZ `.gz` and KataGo `.bin.gz` are third-party assets).
  Put the weights outside the repo and pass them with `-model`.
- When the main model is an LZ weight file, `nnRandomize` is automatically disabled. The LZ genmove
  temperature is also zeroed automatically unless explicitly configured.
- This is not an official KataGo release. Upstream releases and tags belong to
  `lightvector/KataGo`; releases published here are for this fork only.
