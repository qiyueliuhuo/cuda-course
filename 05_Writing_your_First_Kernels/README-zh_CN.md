# 编写你的第一个 CUDA 核函数

> 一切从这里开始 -> https://docs.nvidia.com/cuda/
> 我们主要关注 CUDA C 编程指南 -> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
> 推荐参考这份入门教程 -> https://developer.nvidia.com/blog/even-easier-introduction-cuda/

- 通常建议先在 CPU 上编写核函数代码（更容易编写），再移植到 GPU 上，这样可以确保你的逻辑在线程块（block）和线程（thread）层面保持一致。你可以设置一些输入 x，在两端测试，确保结果一致。

- 手动练习向量加法和矩阵乘法
- 理解线程（thread）、线程块（block）、网格（grid）的概念

## 编译并运行我们的向量加法核函数：

```bash
nvcc -o 01 01_vector_addition.cu
./01
```

（建议结合 assets 文件夹中的简要说明和示意图）

## 硬件资源映射（Hardware Mapping）

- CUDA 核心 (CUDA cores) 负责执行线程
- 流多处理器（Streaming Multiprocessors, SMs）负责管理线程块（一个 SM 通常可以承载多个 block，具体数量取决于资源需求）
- 网格 (Grid) 则映射到整个 GPU，因为它是层次结构中最高一级的单位

## 存储模型（Memory Model）

- 寄存器 (Registers) 和局部内存 (Local Memory)
- 共享内存 (Shared Memory) ⇒ 允许同一个线程块内的线程相互通信
- L2 缓存：作为核心/寄存器与全局内存之间的缓冲区，同时也是跨 SM 的共享内存
- L2 缓存与共享内存（Shared Memory）/L1 缓存都基于 SRAM 架构，因此运行速度相近。L2 缓存更大，作用如下：
- 速度：虽然同为 SRAM，L2 通常比 L1 慢。这并不是底层工艺的原因，而是因为：
  - 容量更大，访问时间增加
  - 多 SM 共享，访问机制更复杂
  - 物理位置更远，距离计算单元更远
- 全局内存（Global Memory）：用于存储从主机复制到设备的数据。设备上的所有程序都可访问全局内存
- 主机（Host）：通常指 CPU 端的内存（如 16/32/64GB DRAM），具体容量取决于你的设备配置
- 如果数组过大，无法完全存储在寄存器中，数据会溢出到局部内存。我们的目标是尽量避免这种情况，以确保程序运行速度最快

![](assets/memhierarchy.png)

### 什么是“随机访问内存”？

- 比如在录影带中，你必须按顺序访问每个位才能到达最后一个。而“随机”指的是可以直接从任意索引获取信息（无需依赖先前索引的数据）。我们获得了一种抽象，表面上看内存像一条巨大的线，但在芯片内部实际上布局为网格结构，只是外部被抽象为线性空间。

![](../assets/memmodel.png)

> [高效矩阵转置：NVIDIA 官方博客](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)