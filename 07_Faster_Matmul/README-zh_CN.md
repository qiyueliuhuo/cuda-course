# 让我们来优化矩阵乘法 (Matrix Multiplication)

![](assets/comparison.png)

> **Naive** (最容易理解，但性能较差)
> **合并内存访问 (Coalesced Memory Access)** (确保以对 GPU 最优的方式加载数据)
> **共享内存 (Shared Memory)** (减少全局内存访问次数以增加有效内存带宽)
> **1D/2D 分块 (Blocktiling)** (将工作均匀地分配给网格中的所有 SM / Block)
> **向量化内存访问 (Vectorized Memory Access)** (单条指令加载更多数据（使用 128 bit 代替 32 bit）)
> **自动调优 (Autotuning)** (针对特定 GPU 架构，通过网格搜索寻找最理想的 Kernel 参数)
> **cuBLAS** (NVIDIA 提供的用于线性代数运算（如 Matmul）的闭源库)

**由于我不想写重复的内容，让我们直接跳到 Simon Boehm 的 [博客](https://siboehm.com/articles/22/CUDA-MMM) 和 [Git 仓库](https://github.com/siboehm/SGEMM_CUDA)**

## 行优先 (Row Major) vs 列优先 (Column Major)

- **cuBLAS** 期望矩阵采用**列优先 (Column Major)** 格式，因此我们需要预先进行转置。
- **行优先 (Row Major)**：`A[i][j]` 存储在 `A[i * N + j]` 中。
- **列优先 (Column Major)**：`A[i][j]` 存储在 `A[j * M + i]` 中。

```python
# 行优先 (Row Major)
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

# 在内存中的存储方式
A = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 列优先 (Column Major)
A = [[1, 4, 7],
     [2, 5, 8],
     [3, 6, 9]]

# 在内存中的存储方式
A = [1, 4, 7, 2, 5, 8, 3, 6, 9]
```

## `pragma #unroll` 的作用

- 理想情况下，你希望每次迭代执行更多有效的计算。如果你能在一次迭代中完成 4 次数学运算而不是 1 次，那是极好的。
- 在某些情况下，即使不显式告知，编译器其实也会自动展开循环（这就是在 `unrolling.cu` 中发生的情况）。
- 你可以使用 `nvcc -ptx v1.cu -o - | less` 查看 **PTX 汇编代码**，以确认编译器是否已经展开了循环。
- 通过编写一个未展开循环的 Kernel 并与展开循环的版本进行性能基准测试，你可以判断循环展开是否真的带来了收益。然后检查 PTX 代码以确认编译器行为。通常只有在你没有获得预期性能并需要深入调查时，这才有必要。
- 为了快速测试，只需记录 Kernel 的平均运行时间并与展开版本对比。如果展开版更快，则说明是有益的。务必验证计算结果的正确性（进行逐元素比较）。

## 什么是占用率 (Occupancy)？

**占用率 (Occupancy)** 定义为每个 SM 的**活跃 Warp (Active Warps)** 数量与每个 SM 能够支持的**最大活跃 Warp** 数量之比。

限制 SM 加载更多活跃 Block 的主要因素有三个：**寄存器数量 (Register count)**、**Warp 数量** 和 **共享内存 (SMEM) 容量**。我们可以为当前的 Kernel 进行示例计算。

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy

> [矩阵乘法性能指南 (Matmul Performance)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

## 汇编指令 (Assembly Instructions):

- [PTX 指令 (Parallel Thread Execution)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-machine-model)
- [如何阅读着色器汇编 (SASS)](https://interplayoflight.wordpress.com/2021/04/18/how-to-read-shader-assembly/)

### 为什么我们可能想要深入研究或编写汇编代码？

- 让我们理解程序受限的具体操作（例如：**Warp 散列 / 分歧 (Warp divergence)**、等待数据到达寄存器、耗时长的昂贵操作等）。
- 允许进行**时钟周期级 (Clock-cycle)** 的优化（这是你能达到的最接近底层硬件的程度）。

## 灵感来源：

1. [Simon Boehm @ Anthropic](https://siboehm.com/articles/22/CUDA-MMM)
2. [Lei Mao @ NVIDIA](https://github.com/leimao/CUDA-GEMM-Optimization)

## 进阶探索：

- 如果想了解 NVIDIA 等公司为了在 cuBLAS 中实现极高的 **TFLOP** 算力而对 **matmul** 采用的 Kernel 性能优化技术，请查看 **cuTLASS** (CUDA Templates for Linear Algebra Subroutines)：
- [CUTLASS Github](https://github.com/NVIDIA/cutlass)
- [CUTLASS Blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [CUTLASS Documentation](https://nvidia.github.io/cutlass/)
