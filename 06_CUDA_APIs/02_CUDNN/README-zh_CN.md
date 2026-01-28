# cuDNN

理论上，编写 GPT 的训练和推理并不一定需要 cuFFT 或大量的定制化 Kernel。cuDNN 中内置了快速卷积（fast convolve），且 cuBLAS 的矩阵乘法（matmul）也被包含在 cuDNN 更高层级的抽象中。不过，了解“慢速卷积 vs 快速卷积”、“慢速矩阵乘法 vs 快速矩阵乘法”的概念仍然大有裨益。

NVIDIA cuDNN 为深度学习应用中频繁出现的各种操作提供了高度优化的实现：

- 前向（Forward）和后向（Backward）卷积，包括互相关（cross-correlation）
- GEMM (通用矩阵乘法)
- 前向和后向池化（Pooling）
- 前向和后向 Softmax
- 前向和后向神经元激活（Neuron activations）：relu, tanh, sigmoid, elu, gelu, softplus, swish。以及算术、数学、关系和逻辑逐点运算（pointwise operations）。
- 张量变换函数（reshape, transpose, concat 等）
- 前向和后向 LRN, LCN, batch normalization (批归一化), instance normalization, 和 layer normalization (层归一化)

除了为单个操作提供高性能实现外，该库还支持灵活的多操作融合（fusion）模式，以进一步优化性能。其目标是为重要的深度学习用例在 NVIDIA GPU 上实现最佳性能。

在 cuDNN 7 及更早版本中，API 设计用于支持一组固定的操作和融合模式。我们非正式地将其称为“传统 API (Legacy API)”。由于流行的融合模式集迅速扩展，从 cuDNN 8 开始，我们添加了 [Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html#graph-api)，它允许用户通过定义操作图（Operation Graph）来表达计算，而不是从一组固定的 API 调用中进行选择。与传统 API 相比，这提供了更好的灵活性，并且对于大多数用例，这是推荐使用 cuDNN 的方式。

你最初可能将“Graph API”一词与图神经网络（Graph Neural Networks）相关的操作混淆。实际上，它只是让你以图（Graph）的形式定义你偏好的操作流。与使用无法看到底层代码的固定操作（传统 API，因为它是预编译的二进制文件）不同，Graph API 允许你在不直接更改底层源代码的情况下进行扩展。

以下是关于 cuDNN 文档的大致思路：

你会看到这些以“不透明结构体类型 (opaque struct types)”实现的张量描述符（tensor descriptor）类型。这些描述符可以创建张量、定义张量操作、获取张量属性等。

我们将对以下代码片段进行逆向工程（你可以将这些术语输入谷歌搜索，找到 Graph API，并粘贴 `cudnnConvolutionForward` 来找到对应的文档，然后映射出周围的所有内容，并更深入地挖掘描述符类型）：

`cudnnTensorDescriptor_t`

`cudnnHandle_t`

`cudnnConvolutionDescriptor_t`

`cudnnFilterDescriptor_t`

`cudnnCreateTensorDescriptor`
`cudnnSetTensor4dDescriptor`

`cudnnConvolutionFwdAlgo_t`

`cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, d_output_cudnn)`

参数说明：包含一个 cuDNN 句柄（handle）、指向 alpha 参数的指针（非描述符类型）、输入描述符（input descriptor）、显存中的卷积输入、卷积滤波器/卷积核描述符（filter/kernel descriptor）、卷积核张量本身、卷积操作描述符（convDesc）、作为前向预测算法类型的 algo（点击此链接后的最上方条目 ⇒ https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-ops-library.html#id172）、GPU 执行卷积操作所需的内存工作空间（workspace 和 workspaceSize）、指向 float 参数的指针 beta、输出描述符以及显存中的输出张量。

你希望 cuDNN 将输入张量视作：

```python
tensor([[[-1.7182,  1.2014, -0.0144],
         [-0.6332, -0.5842, -0.7202]],

        [[ 0.6992, -0.9595,  0.1304],
         [-0.0369,  0.8105,  0.8588]],

        [[-1.0553,  1.9859,  0.9880],
         [ 0.6508,  1.4037,  0.0909]],

        [[-0.6083,  0.4942,  1.9186],
         [-0.7630, -0.8169,  0.6805]]])
```

作为 PyTorch 的参考。但当你分配内存时，它只是一个 int/float 数组。

```python
[-1.7182,  1.2014, -0.0144, -0.6332, -0.5842, -0.7202,  0.6992, -0.9595,
	0.1304, -0.0369,  0.8105,  0.8588, -1.0553,  1.9859,  0.9880,  0.6508,
	1.4037,  0.0909, -0.6083,  0.4942,  1.9186, -0.7630, -0.8169,  0.6805])
```

事实证明，这部分并没有想象中那么糟糕。注意我们的形状是 (4, 2, 3)。我们可以将其分解为 4 个相等的部分（批大小/batch elements），将每一部分再分为 2 个部分（可能是时间维度），此时我们就得到了原始预期的形状。这就是 cuDNN 在底层处理张量的方式。只要你正确指定了形状（例如：**NCHW** ⇒ batch_size, channels, height, width），就不用担心（当然，仍然需要进行 cuDNN 错误检查）。

这里使用的所有代码都在 `01 Conv2d.cu` 中。


1. **预编译单操作引擎 (Pre-compiled Single Operation Engines)**:
    - 这些引擎针对特定的单个操作进行了预编译和优化。由于是预编译的，它们执行效率极高，但在执行的操作方面缺乏灵活性。
    - 示例：专门为矩阵乘法操作预编译并优化的引擎。
2. **通用运行时融合引擎 (Generic Runtime Fusion Engines)**:
    - 这些引擎旨在运行时动态融合多个操作。与预编译引擎相比，它们提供了更多灵活性，因为它们可以适应不同的操作组合，但优化程度可能不如预编译或专门的运行时引擎。
    - 示例：在执行期间动态融合张量上的不同逐元素（element-wise）操作，以避免冗余的内存读写（你可以将不常用的操作融合在一起，获得不错的提升，但仍不如预编译引擎快）。
3. **专用运行时融合引擎 (Specialized Runtime Fusion Engines)**:
    - 类似于通用运行时融合引擎，但针对特定的模式或操作组合进行了专门优化。它们仍然提供运行时灵活性，但也尝试利用针对特定用例或操作序列的优化。
    - 示例：优化了卷积层后接激活函数的引擎。它会在 CUDA 脚本编译期间识别你的代码架构或某些模式，并在后端找到融合操作以获得加速。
4. **专用预编译融合引擎 (Specialized Pre-compiled Fusion Engines)**:
    - 这些引擎针对特定的操作序列进行了预编译和优化。它们提供与预编译单操作引擎一样的高性能，但可以处理一系列操作而不仅仅是单个。
    - 示例：针对神经网络中特定的卷积块（结合了卷积、批量归一化和 ReLU 激活函数）的预编译引擎。

### 运行时融合 (Runtime Fusion):

考虑这样一种场景：你需要对张量执行多个逐元素操作，例如加法、乘法和 sigmoid 激活函数。如果没有运行时融合，每个操作都将是一次单独的 Kernel 启动，且每次都需要读写全局内存（Global Memory）：

`output = torch.sigmoid(tensor1 + tensor2 * tensor3)`

通过运行时融合，上述操作可以合并到一次 Kernel 启动中，从而一次性完成整个计算，将中间结果保存在寄存器（Registers）中，仅将最终输出写入全局内存。

## Graph API

- https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html
- 当然，要让融合变得有意义，图（Graph）需要支持多项操作。理想情况下，我们希望支持的模式足够灵活，能涵盖各种用例。为了实现这种普适性，cuDNN 拥有运行时融合引擎，可以根据图模式在运行时生成一个（或多个）Kernel。本节概述了这些运行时融合引擎（即具有 `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION` 行为注解的引擎）支持的模式。


![](../assets/knlfusion1.png)
![](../assets/knlfusion2.png)

你需要检查 GPU 的计算能力（Compute Compatibility），以查看哪些操作可以融合。

1. Graph API -> Kernel 融合，其中节点（Nodes）是“操作”，边（Edges）是“张量”
2. Ops API -> 单操作引擎（softmax, batchnorm, dropout 等）
3. CNN API -> 卷积及其相关操作（被 Graph API 所依赖）
4. Adversarial API -> “其他”功能和算法（RNN, CTC loss, 多头注意力等）

## 性能基准测试 (Performance Benchmarking)
- 假设你想为你的用例寻找最快的 cuDNN 前向卷积算法：
你会查看算法类型中的不同算法（例如 `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`）并对比每种算法的性能。
- 有时，自己编写 Kernel 比依赖 cuDNN 能获得更好的性能。
- 回顾 cuDNN Graph API，你可以实现自己的操作“图”并将其融合在一起，从而在特定的前向/后向传递过程中实现加速。
- 如果你不对批数据进行处理（non-batch processing），编写自己的优化定制 Kernel 可能会更合适（生产案例）。

## 浏览 cuDNN API
- 只需通过 Ctrl+点击 或 Cmd+点击 函数名即可查看源代码（例如 `cudnnConvolutionForward`）：
```cpp
cudnnConvolutionForward(cudnnHandle_t handle,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y);
```
