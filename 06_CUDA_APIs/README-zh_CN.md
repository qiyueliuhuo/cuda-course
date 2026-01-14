# CUDA API
> 包含 cuBLAS、cuDNN、cuBLASmp

- “API” 这个术语一开始可能有些困惑。它本质上表示：我们使用的是一个库，但看不到其内部实现。API 会提供函数调用的文档说明，但它是预编译的库（二进制），我们只是按照文档去调用这些接口，而不是阅读或修改其源码。

## 不透明结构体类型（Opaque Struct Types，CUDA API）
- 你无法查看或触碰这些类型的内部结构，只能使用它们的外部信息，例如类型名、函数参数等。`.so`（共享对象，shared object）文件作为不透明的二进制被引用，用于以高性能方式直接运行已经编译好的函数。

如果你的目标是在集群上把推理性能做到尽可能快，那么你需要理解这些库在“引擎盖之下”的细节。为了更高效地检索和导航 CUDA API，我建议使用以下途径：
1. [perplexity.ai](http://perplexity.ai)（信息更新最快，且可实时检索）
2. Google 搜索（可能不如 Perplexity，但作为传统方法也完全可用）
3. ChatGPT（用于训练截止前的通用知识，往往仍然有效）
4. 在 NVIDIA 文档中进行关键词搜索

## 错误检查（API 专用）

- 以 cuBLAS 为例

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- 以 cuDNN 为例

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- 错误检查的一般流程如下：先为某个 CUDA API 调用配置上下文，然后发起该操作，随后通过该 API 调用返回的状态码检查操作结果；若失败，则报告并终止，避免错误传播。
- 当然，针对其他 CUDA API 还有更多错误检查宏，但上述是最常用、并且本课程会用到的。
- 建议阅读这篇指南：[Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)

## 矩阵乘法（Matrix Multiplication）
- cuDNN 通过特定的卷积与深度学习算子“隐式”支持矩阵乘法（matmul），但并未把 matmul 作为 cuDNN 的主要特性之一直接呈现。
- 对于矩阵乘法，你通常应优先使用 cuBLAS 提供的深度学习线性代数算子，它覆盖面更广，并针对高吞吐量的 matmul 做过优化调优。
> 旁注（旨在说明：通过配置与调用操作的方式，在不同 CUDA 库之间迁移知识并不困难，例如从 cuDNN 迁移到 cuFFT）。

## 资源
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)