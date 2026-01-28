# 附加内容 (Extras)

## CUDA 编译器
![](assets/nvcc.png)

## CUDA 如何处理条件 if/else 逻辑？
- CUDA 处理条件 if/else 逻辑的效果并不理想。如果你在核函数（Kernel）中使用了条件语句，编译器通常会为两个分支都生成代码，然后使用**谓词指令（predicated instruction）**来选择正确的结果。如果分支很长或者条件很少达成，这会导致大量的计算浪费。因此，通常建议尽可能避免在 Kernel 中使用条件逻辑。
- 如果无法避免，你可以深入查看 PTX 汇编代码（`nvcc -ptx kernel.cu -o kernel`），了解编译器的处理方式。然后，你可以查看所用指令的计算指标，并据此尝试进行优化。
- 当单个线程进入冗长的嵌套 if/else 语句时，其行为会趋向于串行化，这会导致同一 Warp 内的其他线程在等待该线程完成时处于空闲状态。这种现象被称为 **Warp 分歧（warp divergence）**，是 CUDA 编程中处理 Warp 内线程时的常见问题。
- 向量加法之所以快，是因为它不存在产生分歧的可能性，指令执行的路径是唯一的。

## 统一内存 (Unified Memory) 的优缺点
- **统一内存（Unified Memory）**是 CUDA 的一项特性，它允许你分配 CPU（系统 DRAM）和 GPU 都能访问的内存。这可以简化代码中的内存管理，因为你不需要操心在系统 RAM 条和 GPU 显存之间来回手动拷贝数据。
- [CUDA 中的统一内存 vs 显式内存](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
- [最大化统一内存性能](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- 统一内存通过**流（Streams）**自动处理预取（Prefetching）（这也是上述 GitHub 链接中延迟较低的原因）。
    - [CUDA 流 - Lei Mao](https://leimao.github.io/blog/CUDA-Stream/)
    - [NVIDIA 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
    - 流允许数据传输（预取）与计算相互重叠。
    - 当一个流正在执行 Kernel 时，另一个流可以为下一次计算传输数据。
    - 这种技术通常被称为**“双缓冲”（double buffering）**，如果扩展到更多缓冲区，则称为“多缓冲”。

![](assets/async.png)

## 内存架构 (Memory Architectures)
- **DRAM/VRAM 单元**是计算机中最小的内存单位。它们由存储数据位的电容器和晶体管组成。电容器以电荷形式存储位，而晶体管控制电流流向以读取和写入数据。
- ![](assets/dram-cell.png)
- **SRAM (共享内存)** 是一种比 DRAM 更快但更昂贵的内存。它被用作 CPU 和 GPU 的缓存内存（Cache），因为它的访问速度远高于 DRAM。
- 现代 NVIDIA GPU 的大多数片上存储（on-chip memory）可能使用 **6T**（六晶体管）或 **8T** SRAM 单元。
- 6T 单元结构紧凑且性能良好，而 8T 单元可以提供更好的稳定性和更低的功耗，但代价是占用面积更大。
- NVIDIA GPU 在不同架构和计算能力（Compute Capability）中具体如何使用 6T vs 8T SRAM 单元并未公开详细信息。与大多数半导体公司一样，NVIDIA 对许多底层设计选择保持私有。
- ![](assets/sram-cell.png)
- ![](assets/8t-sram-cell.png)


## 深入探索 (Dive deeper)
- 量化 (quantization) -> fp32 -> fp16 -> int8
- Tensor Cores (WMMA)
- 稀疏性 (sparsity) -> [0, 0, 0, 0, -7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
- [CUDA by Example (英文版)](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf)
- [深度学习模型的数据并行分布式训练](https://siboehm.com/articles/22/data-parallel-training)
- [mnist-cudnn 项目](https://github.com/haanjack/mnist-cudnn)
- [CUDA MODE 讲座](https://github.com/cuda-mode/lectures)
- [micrograd-cuda 项目](https://github.com/mlecauchois/micrograd-cuda)
- [micrograd 项目](https://github.com/karpathy/micrograd)
- [GPU 谜题 (GPU puzzles)](https://github.com/srush/GPU-Puzzles)
