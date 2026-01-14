# GPU简介（入门篇）

> 本章旨在介绍GPU的发展历史、为什么在深度学习任务中使用 GPU，以及为什么 GPU 在某些任务上比 CPU 快得多。

## 硬件介绍
![](assets/cpu.png)
- CPU：中央处理器（Central Processing Unit）
    - 通用型
    - 高主频
    - 少量核心
    - 大缓存
    - 低延迟
    - 低吞吐量

![](assets/gpu.png)
- GPU：图形处理器（Graphics Processing Unit）
    - 专用型
    - 低主频
    - 大量核心
    - 小缓存
    - 高延迟
    - 高吞吐量

![](assets/tpu.png)
- TPU：张量处理器（Tensor Processing Unit）
    - 专为深度学习算法（如矩阵乘法等）设计的专用GPU

![](assets/fpga.png)
- FPGA：现场可编程门阵列（Field Programmable Gate Array）
    - 可重新配置硬件以执行特定任务
    - 极低延迟
    - 极高吞吐量
    - 能耗极大
    - 价格极高

## NVIDIA GPU 简史
> NVIDIA GPU简史视频推荐 -> https://www.youtube.com/watch?v=kUqkOAU84bA

![](assets/history01.png)
![](assets/history02.png)
![](assets/history03.png)

## 为什么 GPU 在深度学习领域如此高效？
![](assets/cpu-vs-gpu.png)

- CPU（主机端）
    - 目标是最小化单个任务的耗时
    - 衡量指标：延迟（单位：秒）

- GPU（设备端）
    - 目标是最大化整体吞吐量
    - 衡量指标：吞吐量（单位：每秒处理的任务数量，例如每毫秒处理的像素数）

## 典型的CUDA程序流程
1. CPU分配主机内存
2. CPU将数据复制到GPU
3. CPU在GPU上启动内核（实际计算在此处进行）
4. CPU将结果从GPU拷贝回主机，并做进一步处理

内核看起来像是串行程序，并没有直接体现并行性。可以想象你正在拼一副拼图，而你已经知道每一块拼图的位置。高级算法的设计就是让你针对每一块单独解决问题：“把这块拼图放到正确的位置”。只要最后所有拼块都拼在正确的位置，整体就完成了！你无需从某个角落开始逐步拼完整个拼图。只要各块之间互不干扰，你可以同时处理多块拼图。

## 需要记住的一些术语
- 内核（kernel）：这里指 GPU 内核，不是爆米花，也不是卷积核，更不是Linux内核
- 线程、块、网格（下一章详细介绍）
- GEMM = **GE**neral **M**atrix **M**ultiplication（广义矩阵乘法）
- SGEMM = **S**ingle precision (fp32) **GE**neral **M**atrix **M**ultiplication（单精度广义矩阵乘法）
- CPU/主机/主机函数 vs GPU/设备/内核函数
- CPU通常被称为主机（host），它负责执行普通函数。
- GPU通常被称为设备（device），它负责执行被称为内核（kernel）的GPU专用函数。