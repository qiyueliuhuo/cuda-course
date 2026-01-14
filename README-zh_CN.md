# CUDA 课程

本仓库为 FreeCodeCamp 的 CUDA 课程的 GitHub 仓库。

> 注：本课程面向 Ubuntu Linux 设计。Windows 用户可使用 Windows Subsystem for Linux（WSL）或 Docker 容器来模拟 Ubuntu Linux 环境。

## 目录

1. [深度学习生态系统](01_Deep_Learning_Ecosystem/README-zh_CN.md)
2. [环境搭建 / 安装](02_Setup/README-zh_CN.md)
3. [C/C++ 复习](03_C_and_C++_Review/README-zh_CN.md)
4. [GPU 简介](04_Gentle_Intro_to_GPUs/README-zh_CN.md)
5. [编写你的第一个核函数（Kernels）](05_Writing_your_First_Kernels/README-zh_CN.md)
6. [CUDA API（cuBLAS、cuDNN 等）](06_CUDA_APIs/README-zh_CN.md)
7. [优化矩阵乘法](07_Faster_Matmul/README-zh_CN.md)
8. [Triton](08_Triton/README-zh_CN.md)
9. [PyTorch 扩展（CUDA）](08_PyTorch_Extensions/README-zh_CN.md)
10. [期末项目](09_Final_Project/README-zh_CN.md)
11. [扩展内容](10_Extras/README-zh_CN.md)

## 课程理念

本课程旨在：

- 降低进入高性能计算（HPC）工作的门槛
- 为理解诸如 Karpathy 项目 [llm.c](https://github.com/karpathy/llm.c) 等打下基础
- 将分散的 CUDA 编程资源整理为一门全面、有条理的课程

## 概览

- 侧重于通过 GPU kernel 优化来提升性能
- 涵盖 CUDA、PyTorch 与 Triton
- 强调编写高效核函数的技术细节
- 面向 NVIDIA GPU
- 课程最终将完成一个基于 CUDA 的简单 MLP（多层感知机）MNIST 项目

## 前置知识

- 需要具备 Python 编程能力（必须）
- 了解基本微分和向量微积分，用于反向传播（推荐）
- 掌握线性代数基础（推荐）

## 关键收获

- 优化现有实现
- 为前沿研究构建 CUDA 核函数
- 理解 GPU 性能瓶颈，尤其是内存带宽相关瓶颈

## 硬件需求

- 任何 NVIDIA 的 GTX、RTX 或数据中心级别 GPU
- 无本地硬件者可使用云端GPU服务

## CUDA / GPU 编程的应用场景

- 深度学习（本课程的重点）
- 图形与光线追踪
- 流体仿真
- 视频编辑
- 加密货币挖矿
- 3D 建模
- 任何需要对大规模数组进行并行处理的任务

## 资源

- GitHub 仓库（本仓库）
- Stack Overflow
- NVIDIA 开发者论坛
- NVIDIA 与 PyTorch 文档
- 使用大型语言模型（LLMs）辅助学习
- 速查表（Cheatsheet）[在此](/11_Extras/assets/cheatsheet.md)

## 其他学习资料

- https://github.com/CoffeeBeforeArch/cuda_programming
- https://www.youtube.com/@GPUMODE
- https://discord.com/invite/gpumode

## 有趣的 YouTube 视频
- [GPU 是如何工作的？探索 GPU 架构](https://www.youtube.com/watch?v=h9Z4oGN89MU)
- [GPU 到底是怎么工作的？](https://www.youtube.com/watch?v=58jtf24uijw&ab_channel=Graphicode)
- [为 Python 程序员入门 CUDA](https://www.youtube.com/watch?v=nOxKexn3iBo&ab_channel=JeremyHoward)
- [从原子开始解释 Transformer（原理）](https://www.youtube.com/watch?v=7lJZHbg0EQ4&ab_channel=JacobRintamaki)
- [CUDA 编程是如何工作的 — Stephen Jones，NVIDIA CUDA 架构师](https://www.youtube.com/watch?v=QQceTDjA4f4&ab_channel=ChristopherHollinworth)
- [使用并行计算与 Nvidia CUDA — NeuralNine](https://www.youtube.com/watch?v=zSCdTOKrnII&ab_channel=NeuralNine)
- [CPU vs GPU vs TPU vs DPU vs QPU](https://www.youtube.com/watch?v=r5NQecwZs1A&ab_channel=Fireship)
- [Nvidia CUDA 100 秒速览](https://www.youtube.com/watch?v=pPStdjuYzSI&ab_channel=Fireship)
- [AI 发现更快矩阵乘法算法的故事](https://www.youtube.com/watch?v=fDAPJ7rvcUw&t=1s&ab_channel=QuantaMagazine)
- [最快的矩阵乘法算法](https://www.youtube.com/watch?v=sZxjuT1kUd0&ab_channel=Dr.TreforBazett)
- [从零开始：CUDA 中的缓存分块（Cache Tiled）矩阵乘法](https://www.youtube.com/watch?v=ga2ML1uGr5o&ab_channel=CoffeeBeforeArch)
- [从零开始：CUDA 中的矩阵乘法](https://www.youtube.com/watch?v=DpEgZe2bbU0&ab_channel=CoffeeBeforeArch)
- [GPU 编程入门](https://www.youtube.com/watch?v=G-EimI4q-TQ&ab_channel=TomNurkkala)
- [CUDA 编程](https://www.youtube.com/watch?v=xwbD6fL5qC8&ab_channel=TomNurkkala)
- [CUDA 入门（第1部分）：高层概念](https://www.youtube.com/watch?v=4APkMJdiudU&ab_channel=JoshHolloway)
- [GPU 硬件入门](https://www.youtube.com/watch?v=kUqkOAU84bA&ab_channel=TomNurkkala)

## 联系我

- [Twitter / X](https://x.com/elliotarledge)
- [LinkedIn](https://www.linkedin.com/in/elliot-arledge-a392b7243/)
- [YouTube](https://www.youtube.com/channel/UCjlt_l6MIdxi4KoxuMjhYxg)
- [Discord](https://discord.gg/JTTcFe7Pw2)