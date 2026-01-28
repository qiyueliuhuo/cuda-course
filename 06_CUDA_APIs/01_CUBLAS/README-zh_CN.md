> 注意：在开始之前，请务必注意，进行预热（warmup）和基准测试（benchmark）运行是准确测量函数执行时间的必要手段。如果不进行任何预热，cuBLAS 在首次运行时会产生巨大的开销（约为 45ms），从而导致结果偏差。基准测试运行旨在获取更准确的平均执行时间。

# cuBLAS

- **NVIDIA CUDA Basic Linear Algebra Subprograms (cuBLAS)** 是一个专为加速 AI 和 HPC（高性能计算）应用设计的 GPU 加速库。它包含了多个 API 扩展，提供了符合行业标准的 BLAS API，以及针对 NVIDIA GPU 高度优化的、支持算子融合（fusion）的 GEMM（通用矩阵乘法）API。
<br>
- 请留意数据形状与内存布局（shaping / memory layout）问题 (参考: https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)

## cuBLAS-Lt
- **cuBLASLt (cuda BLAS Lightweight)** 是 cuBLAS 库的轻量级扩展，提供了更灵活的 API，主要旨在提高特定工作负载（如深度学习模型）的性能。几乎所有的数据类型和 API 调用都围绕着 **matmul**（矩阵乘法）展开。

- 在单个 Kernel 无法处理整个问题的情况下，cuBLASLt 会尝试将问题分解为多个子问题，并通过在每个子问题上运行 Kernel 来解决。

- 这也是 fp16 / fp8 / int8 等混合精度计算发挥威力的地方。

## cuBLAS-Xt
- **cuBLAS-Xt** 用于 **Host + GPU** 联合求解（通常速度会慢很多）。
- 由于需要在系统内存（DRAM）和显存（VRAM）之间进行数据传输，会产生内存带宽瓶颈，计算速度无法达到全在显存片上（on-chip）进行的速度。
- **cuBLASXt** 是 cuBLAS 的扩展，旨在支持 **Multi-GPU**（多 GPU）。**关键特性包括：**

**多 GPU 支持 (Multiple GPUs)：** 能够跨多个 GPU 运行 BLAS 操作，实现 GPU 扩展，并显著提升大规模数据集的性能。

**线程安全 (Thread Safety)：** 采用线程安全设计，支持在不同 GPU 上并发执行多个 BLAS 操作。

**适用场景：** 非常适合可以通过将工作负载分布到多个 GPU 上来获益的大规模计算。

对于超出单张 GPU 显存限制的大规模线性代数计算，请选择 **XT**。

- cuBLAS vs cuBLAS-Xt
    - `(M, N) @ (N, K)` 其中 M = N = K = 16384
    - ![](../assets/cublas-vs-cublasxt.png)

## cuBLASDx

**重点说明：本课程中不涉及/不使用此部分**

**cuBLASDx (preview)** 库是一个设备端（Device-side）API 扩展，用于在 CUDA Kernel 内部执行 BLAS 计算。通过融合数值操作（fusing numerical operations），您可以降低延迟并进一步提高应用程序的性能。

- 您可以点击[此处](https://docs.nvidia.com/cuda/cublasdx)访问 cuBLASDx 文档。
- cuBLASDx 不包含在 CUDA Toolkit 中。您可以从[此处](https://developer.nvidia.com/cublasdx-downloads)单独下载 cuBLASDx。

## CUTLASS
- cuBLAS 及其变体运行在 Host 端，而目前 cuBLAS-DX 的文档不够详尽，优化程度也略逊一筹。
- 矩阵乘法是深度学习中最重要的操作，但 cuBLAS 不支持在 Kernel 层面轻松地将多个操作融合在一起。
- 另一方面，[CUTLASS](https://github.com/NVIDIA/cutlass) (CUDA Templates for Linear Algebra Subroutines) 则提供了这种能力（这部分内容将在选修章节中涵盖）。
- 顺便提一下，**Flash Attention** 并没有使用 CUTLASS，而是使用了深度优化的原生 CUDA Kernel（详见下文）。
![](../assets/flashattn.png) -> 来源: https://arxiv.org/pdf/2205.14135
