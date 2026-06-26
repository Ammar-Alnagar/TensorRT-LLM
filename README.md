<div align="center">

Trinity LLM
===========================
<h4>Trinity LLM optimizes inference for LLMs and Visual Gen models with specialized C++ kernels for common operations, an efficient runtime, and a C++-first framework.</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/downloads/release/python-3123/)
[![python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-13.1.1-green)](https://developer.nvidia.com/cuda-downloads)
[![torch](https://img.shields.io/badge/torch-2.10.0-green)](https://pytorch.org)
[![version](https://img.shields.io/badge/release-1.3.0rc20-green)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE)

[Architecture](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Performance](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://nvidia.github.io/TensorRT-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3Aroadmap)

---
<div align="left">

## Trinity LLM Overview

Trinity LLM is an open-source C++-first library for optimizing LLM and Visual Gen inference. It provides state-of-the-art optimizations, including custom C++ and CUDA kernels for common inference operations (attention, GEMMs, MoE), algorithmic runtime optimizations (Prefill-Decode disaggregation, Wide Expert Parallelism, Speculative Decoding), and much more, to perform inference efficiently on NVIDIA GPUs.

Built on a C++ core with Python bindings, Trinity LLM provides a high-level Python [LLM API](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#llm-api) that supports a wide range of inference setups — from single-GPU to multi-GPU or multi-node deployments. It includes built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) and the [Triton Inference Server](https://github.com/triton-inference-server/server).

Trinity LLM is designed with a **C++-first** philosophy — the majority of performance-critical code lives in `cpp/`, with Python bindings (nanobind) providing the high-level interface. Several popular models are pre-defined and can be customized using native PyTorch or C++ code.

## Getting Started

To get started with Trinity LLM, visit our documentation:

- [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
    - [Running DeepSeek](./examples/models/core/deepseek_v3)
- [Installation Guide](https://nvidia.github.io/TensorRT-LLM/installation/index.html)
- [Supported Hardware, Models, and other Software](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [Benchmarking Performance](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html#benchmarking-with-trtllm-bench)
- [Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)

## Architecture

Trinity LLM follows a **C++-first** architecture:

| Layer | Language | Description |
|-------|----------|-------------|
| **CUDA Kernels** | C++/CUDA | GPU kernels for attention, GEMM, MoE, sampling |
| **Runtime** | C++ | Scheduler, Batch Manager, KV Cache Manager, Decoder |
| **Bindings** | C++ (nanobind) | Python bindings for C++ runtime components |
| **Model Definitions** | C++ / Python | Model architectures (increasingly C++) |
| **LLM API** | Python | High-level user-facing API |

### Key C++ Components

- **`cpp/tensorrt_llm/runtime/`**: Scheduler, decoder, buffer management, IPC utilities
- **`cpp/tensorrt_llm/kernels/`**: CUDA kernels for attention, MoE, sampling, and fused ops
- **`cpp/tensorrt_llm/batch_manager/`**: In-flight batching, KV cache management
- **`cpp/tensorrt_llm/nanobind/`**: Python bindings for all C++ components
- **`cpp/tensorrt_llm/thop/`**: Torch custom ops implemented in C++

### Python → C++ Migration

Trinity LLM is actively migrating from Python to C++. Utility modules are being ported first:

- `tensorrt_llm/math_utils.py` → C++ nanobind utilities
- `tensorrt_llm/version.py` → C++ version constant
- `tensorrt_llm/scheduling_params.py` → C++ struct

The goal is to have the majority of performance-critical and utility code in C++, using Python only for the high-level API layer.

## C++ Development

### Building C++ Components

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Running C++ Tests

```bash
cd cpp/build
ctest --output-on-failure
```

## C++ Code Guidelines

- Follow the `.clang-format` and `.clang-tidy` configurations in the repository root
- All new C++ code must include the NVIDIA copyright header
- Use nanobind for Python bindings (not pybind11)
- Follow RAII principles and avoid raw pointers where possible

## Deprecation Policy

Deprecation is used to inform developers that some APIs and tools are no longer recommended for use. Beginning with version 1.0, Trinity LLM has the following deprecation policy:

1. Communication of Deprecation
   - Deprecation notices are documented in the Release Notes.
   - Deprecated APIs, methods, classes, or parameters include a statement in the source code indicating when they were deprecated.
   - If used, deprecated methods, classes, or parameters issue runtime deprecation warnings.
2. Migration Period
   - Trinity LLM provides a 3-month migration period after deprecation.
   - During this period, deprecated APIs, tools, or parameters continue to work but trigger warnings.
3. Scope of Deprecation
   - Full API/Method/Class Deprecation: The entire API/method/class is marked for removal.
   - Partial Deprecation: If only specific parameters of an API/method are deprecated, the method itself remains functional, but the deprecated parameters will be removed in a future release.
4. Removal After Migration Period
   - After the 3-month migration period ends, deprecated APIs, tools, or parameters are removed in a manner consistent with semantic versioning.

## Telemetry Data Collection

Trinity LLM collects anonymous telemetry data by default. This data is used
in aggregate to understand usage patterns and prioritize engineering efforts.
**This data cannot be traced back to any individual user.** No prompts,
outputs, model weights, model paths, tokenizer paths, user-identifying
information, raw free-form configuration strings, or persistent identifiers are
collected. Any deployment identifiers are ephemeral, randomly generated per
deployment, and not linked to users. The data we collect includes:

- Ingress point (e.g., LLM API, CLI, serve command)
- Deployment duration (via periodic heartbeats)
- GPU SKUs, count, memory, and CUDA version
- Model architecture class name (e.g., `LlamaForCausalLM`)
- Parallelism configuration (TP/PP/CP/MoE-EP/MoE-TP sizes), quantization algorithm, dtype, KV cache dtype
- System information (OS platform, Python version, CPU architecture, CPU count)
- Trinity LLM version and backend
- Feature summary flags (LoRA, speculative decoding, prefix caching, CUDA graphs, chunked context, data parallelism)
- Disaggregated serving metadata (role and deployment ID)
- Selected LLM API configuration values: parallelism, dtype, KV cache, scheduler, CUDA graph, and compile settings
- Capture diagnostics for that payload: a schema checksum (for provenance), the count of captured fields, and whether any free-form value was skipped

Telemetry is automatically disabled in CI and test environments.

### Opting Out of Telemetry Data Collection

To disable telemetry data collection, use any of the following methods:

- **Environment variable**: Set `TRTLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, or `TELEMETRY_DISABLED=true`
- **File-based**: Create the file `~/.config/trtllm/do_not_track`
- **Python API**: Pass `TelemetryConfig(disabled=True)` to `LLM()`
- **CLI flag**: Use `--no-telemetry` on `trtllm-serve`, `trtllm-bench`, or `trtllm-eval`

The telemetry collection code is fully open source and auditable at
[`tensorrt_llm/usage/`](./tensorrt_llm/usage/). For a detailed field-by-field
reference of exactly what is collected, see the
[schema documentation](./tensorrt_llm/usage/schemas/README.md).

## Useful Links
- [Quantized models on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4): A growing collection of quantized (e.g., FP8, FP4) and optimized LLMs, including [DeepSeek FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4), ready for fast inference with Trinity LLM.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo): A datacenter scale distributed inference serving framework that works seamlessly with Trinity LLM.
- [WeChat Discussion Group](https://github.com/NVIDIA/TensorRT-LLM/issues/5359): A real-time channel for Trinity LLM Q&A and news.
