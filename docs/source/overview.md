(product-overview)=

# Overview

## About Trinity LLM

[Trinity LLM](https://developer.nvidia.com/tensorrt) is NVIDIA's comprehensive open-source C++-first library for accelerating and optimizing inference performance of the latest large language models (LLMs) on NVIDIA GPUs.

## Key Capabilities

### 🔥 **C++-First Architecture**

Trinity LLM provides a high-level Python [LLM API](./quick-start-guide.md#run-offline-inference-with-llm-api) built on a performant C++ core. It supports a wide range of inference setups - from single-GPU to multi-GPU or multi-node deployments. It includes built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) and the [Triton Inference Server](https://github.com/triton-inference-server/server).

Trinity LLM is designed with a **C++-first** philosophy - performance-critical code lives in `cpp/`, with Python bindings (nanobind) providing the high-level interface. Several popular models are pre-defined and can be customized using native C++ code.

### ⚡ **State-of-the-Art Performance**

Trinity LLM delivers breakthrough performance on the latest NVIDIA GPUs with optimized C++ and CUDA kernels:

- **DeepSeek R1**: World-record inference performance on Blackwell GPUs
- **Llama 4 Maverick**: Breaks the 1,000 TPS/User Barrier on B200 GPUs

### 🎯 **Comprehensive Model Support**

Trinity LLM supports the latest and most popular LLM and DiT architectures. See [complete list](./models/supported-models.md).

- **Language Models**: GPT-OSS, Deepseek-R1/V3, Llama 3/4, Qwen2/3, Gemma 3, Phi 4...
- **Multi-modal Models**: LLaVA-NeXT, Qwen2-VL, VILA, Llama 3.2 Vision...
- **[Visual Generation](./models/visual-generation.md) Models**: FLUX, Wan2.1/2.2 for image and video generation.

### FP4 and FP8 Support

Trinity LLM leverages NVIDIA's latest hardware capabilities with native FP4 and FP8 quantization support, enabling efficient low-precision inference through optimized C++ and CUDA kernels.

### 🚀 **Advanced Optimization & Production Features**
- **[In-Flight Batching & Paged Attention](./features/paged-attention-ifb-scheduler.md)**: In-flight batching eliminates wait times by dynamically managing request execution, processing context and generation phases together for maximum GPU utilization and reduced latency.
- **[Multi-GPU Multi-Node Inference](./features/parallel-strategy.md)**: Seamless distributed inference with tensor, pipeline, and expert parallelism across multiple GPUs and nodes.
- **[Advanced Quantization](./features/quantization.md)**: FP4 and FP8 quantization with optimized C++ kernels.
- **[Speculative Decoding](./features/speculative-decoding.md)**: Multiple algorithms including EAGLE, MTP and NGram.
- **[KV Cache Management](./features/kvcache.md)**: Paged KV cache with intelligent block reuse and memory optimization.
- **[Chunked Prefill](./features/paged-attention-ifb-scheduler.md)**: Efficient handling of long sequences by splitting context into manageable chunks.
- **[LoRA Support](./features/lora.md)**: Multi-adapter support with HuggingFace and NeMo formats.
- **[Checkpoint Loading](./features/checkpoint-loading.md)**: Flexible model loading from various formats.
- **[Guided Decoding](./features/guided-decoding.md)**: Advanced sampling with stop words, bad words, and custom constraints.
- **[Disaggregated Serving (Beta)](./features/disagg-serving.md)**: Separate context and generation phases across different GPUs for optimal resource utilization.

## C++ Core Components

| Component | Path | Description |
|-----------|------|-------------|
| Runtime | `cpp/tensorrt_llm/runtime/` | Scheduler, decoder, buffer management, IPC |
| Kernels | `cpp/tensorrt_llm/kernels/` | CUDA kernels for attention, MoE, sampling |
| Batch Manager | `cpp/tensorrt_llm/batch_manager/` | In-flight batching, KV cache management |
| Nanobind | `cpp/tensorrt_llm/nanobind/` | Python bindings for all C++ components |

## What Can You Do With Trinity LLM?

Whether you're building the next generation of AI applications, optimizing existing LLM deployments, or exploring the frontiers of large language model technology, Trinity LLM provides the tools, performance, and flexibility you need to succeed in the era of generative AI. To get started, refer to the {ref}`quick-start-guide`.
