# Triton

## 设计 (Design)

- **CUDA** -> 标量程序 (scalar program) + 分块线程 (blocked threads)
- **Triton** -> 分块程序 (blocked program) + 标量线程 (scalar threads)

![](../assets/triton1.png)
![](../assets/triton2.png)

### 分块程序 + 标量线程 (Triton) vs 标量程序 + 分块线程 (CUDA)

- **CUDA 被认为是“标量程序 + 分块线程”**：因为我们编写的核函数（Kernel）是在线程（标量）级别进行操作的，而 **Triton 则被抽象到了线程块（Thread Blocks）级别**（编译器会为我们处理线程级别的具体操作）。
- **CUDA 具有“分块线程”的特性**：我们需要在块（Block）级别“操心”线程间的同步与通信；而 **Triton 具有“标量线程”的特性**：我们不需要在线程级别“操心”线程间的问题（编译器同样会代劳）。

### 这种设计在直观层面意味着什么？

- 为深度学习操作（如激活函数、卷积、矩阵乘法 matmul 等）提供了更高级别的抽象。
- 编译器会自动处理复杂的样板代码（boilerplate），如加载（load）和存储（store）指令、分块 (Tiling)、SRAM 缓存等。
- Python 程序员也能编写出性能足以媲美 cuBLAS、cuDNN 的核函数（这对于大多数 CUDA/GPU 程序员来说是非常困难的）。

### 既然如此，我们可以跳过 CUDA 直接学习 Triton 吗？

- **不能**。Triton 是建立在 CUDA 之上的抽象。
- 在很多场景下，你可能仍然需要直接使用 CUDA 来深度优化自己的核函数。
- 你需要理解 CUDA 的编程范式及相关核心概念，才能真正理解如何在 Triton 之上进行构建。

> **相关资源**: [论文 (Paper)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [官方文档 (Docs)](https://triton-lang.org/main/index.html), [OpenAI 相关博文](https://openai.com/index/triton/), [Github 仓库](https://github.com/triton-lang/triton)
