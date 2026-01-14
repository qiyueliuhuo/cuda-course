# 第01章：当前深度学习生态系统

### **免责声明：** 

本部分不会涉及任何与 CUDA 相关的高技术内容。这里更适合向你展示生态系统，而不是盲目进入技术细节。根据我的学习经验，在涉足细节之前，先对整体生态系统有一个大致的认识会更有帮助。否则，学习将变得混乱且容易迷失方向。

随着我们继续深入细节，我建议你对感兴趣的内容主动查阅和尝试（你会在这一章节遇到不少有趣的东西）。如果只是被动听别人讲解，可能会收获有限。主动探索才是学习的关键。

## 研究
- PyTorch ([PyTorch - Fireship](https://www.youtube.com/watch?v=ORMx45xqWkA&t=0s&ab_channel=Fireship))
    - 如果你正在看这部分内容，我假设你已具备 PyTorch 的基础知识。如果没有，建议先看 [Daniel Bourke 的 PyTorch 教程](https://www.youtube.com/watch?v=Z_ikDlimN6A)
    - PyTorch 有 nightly（每日构建版）与 stable（稳定版）可选 ⇒ https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633
        
        每日构建版功能更前沿，但可能不稳定；稳定版更可靠但更新慢
        
    - 用户更喜欢 PyTorch，因为它与 Huggingface 兼容性好
    - 你可以在 torchvision（通过 pip install torchvision 安装）和 torch.hub 上找到预训练模型。PyTorch 生态系统采用了更为去中心化的方式来获取预训练模型，这种方式虽然灵活，但查找起来稍微复杂一些。很多人会把自己的模型发布在 GitHub 仓库，而不是集中上传到统一的模型数据库。由于社区的积极推动，Huggingface 目前是最常用的平台
    - ONNX 支持良好
- TensorFlow ([TensorFlow - Fireship](https://www.youtube.com/watch?v=i8NETqtGHms))
    - 文档完善，社区活跃，是使用最广泛的深度学习框架
    - 相比其他框架推理速度较慢
    - 由 Google 开发（为 TPU 设计），也支持传统 ML（SVM、决策树等）
    - 可在 https://www.tensorflow.org/resources/models-datasets 获取预训练模型
    - 下载预训练模型便捷，1-3 行代码即可
    - ONNX 支持有限（`tf2onnx`）
- Keras
    - 类似于 TensorFlow 里的 torch.nn，但属于更高层的接口
    - 虽然是独立的库，但与 TensorFlow 深度集成，是其主要的高级 API
    - 完整框架，支持模块化开发和训练，而且不仅限于神经网络
- JAX ([JAX - Fireship](https://www.youtube.com/watch?v=_0D5lXDjNpw))
    - JIT 编译（运行时编译）的自动微分加速线性代数
    - 文档地址 ⇒ https://jax.readthedocs.io/en/latest/
    - 编程体验类似 numpy
    - Reddit 上 JAX 讨论 ⇒ https://www.reddit.com/r/MachineLearning/comments/1b08qv6/d_is_it_worth_switching_to_jax_from/
    - JAX 与 TensorFlow 均由 Google 开发
    - 使用 XLA（加速线性代数）编译器
    - 支持 `tf2onnx`
- MLX
    - 由 Apple 针对 Apple Silicon 开发
    - 开源框架
    - 专注于苹果设备上的高性能机器学习
    - 支持训练与推理
    - 针对 Apple Metal GPU 架构优化
    - 支持动态图计算
    - 适合新模型的研发与实验
- PyTorch Lightning
    - https://www.reddit.com/r/deeplearning/comments/t31ppy/for_what_reason_do_you_or_dont_you_use_pytorch/
    - 主要用于减少样板代码和分布式训练的扩展
    - 训练流程简化为 `Trainer()`，无需手动写训练循环

## 生产部署
- 仅推理
    - vLLM
        - ‣
    - TensorRT
        - 可与 PyTorch 无缝集成用于推理
        - 支持 ONNX 格式模型加载
        - 针对 CUDA 内核优化，设计时考虑以下因素：
            - 利用稀疏性的优势
            - 推理量化
            - 硬件架构
            - 显存（VRAM）与片上内存（on-chip memory）之间的内存访问模式
        - Tensor RunTime 的缩写
        - 由 NVIDIA 开发、设计和维护
        - 专为大型语言模型（LLM）推理而构建
        - 本课程涉及其中一些技术，但为了易用性进行了封装和抽象
        - 通常需要 ONNX 模型支持，建议查阅相关文档
        - 参考资料：
            - https://nvidia.github.io/TensorRT-LLM/
            - https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
            - https://pytorch.org/TensorRT/getting_started/installation.html#installation
        - 
- Triton
    - OpenAI 开发与维护 ⇒ https://openai.com/index/triton/
    - ‣
    - 类似 CUDA，但为 Python 设计，简化了传统 CUDA C/C++ 的内核开发流程，矩阵乘法性能可媲美 CUDA
    - 入门指引 ⇒ https://triton-lang.org/main/index.html
    - Triton 内核开发教程 ⇒ https://triton-lang.org/main/getting-started/tutorials/index.html
    - Triton 推理服务器
        - https://developer.nvidia.com/triton-inference-server
        - ‣
        - 
    - 原始论文：https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
    - triton-viz 是 Triton 主要的性能分析和可视化工具
    - 用 Python 精细控制 GPU 行为，无需担心 C/C++ 的复杂性
        - 不需要显式内存管理（`cudaMalloc`、`cudaMemcpy`、`cudaFree`）
        - 不需要错误检查宏（`CUDA_CHECK_ERROR`）
        - 网格/块/线程级索引简化，减少内核启动参数复杂度

![](../05_Writing_your_First_Kernels/assets/triton1.png)
    
- torch.compile
    - 比 TorchScript 更受关注，通常性能更优
    - 将模型编译为静态表示，避免 PyTorch 动态计算图的性能瓶颈，可作为优化后的二进制运行
    - 参考讨论：https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- TorchScript
    - 某些场景下速度更快，尤其是 C++ 部署
    - 性能提升与特定网络结构相关
    - 参考讨论：https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- ONNX Runtime
    - https://youtu.be/M4o4YRVba4o
    - “**ONNX Runtime 训练** 可用一行代码加速多节点 NVIDIA GPU 上的 transformer 训练”
    - 微软开发与维护
- Detectron2
    - 支持训练和推理
    - Facebook（Meta）发起的计算机视觉项目
    - 主要用于检测与分割任务

## 底层框架
- CUDA
    - Compute unified device architecture（CUDA，统一设备计算架构）可理解为 Nvidia GPU 的编程语言。
    - 相关库：cuDNN、cuBLAS、cutlass（快速线性代数与深度学习算法）、cuFFT（快速卷积 FFT，课程会讲到）
    - 可根据硬件架构自定义内核（Nvidia 内部也会通过编译器特殊参数实现优化）
- ROCm
    - AMD GPU 的 CUDA 等价方案
- OpenCL
    - 开放计算语言
    - 支持 CPU、GPU、数字信号处理器等多种硬件
    - Nvidia 设计了 CUDA，因此在 Nvidia 平台上性能优于 OpenCL。如果做嵌入式系统开发，OpenCL 依然值得学习

## 边缘计算与嵌入式系统推理

- 边缘计算指的是在实际分布式系统（如车队）中实现低延迟高效的本地计算。特斯拉 FSD 就是边缘计算的典型应用，因为它需要在本地高效处理数据。

- CoreML
    - 主要用于苹果设备上的预训练模型部署
    - 针对设备端推理优化
    - 支持设备端训练
    - 支持多种模型类型（视觉、自然语言、语音等）
    - 与苹果生态（iOS、macOS、watchOS、tvOS）集成良好
    - 强调隐私，数据留在本地
    - 支持模型从其他框架转换
    - 便于开发者在应用中集成 ML 功能
- PyTorch Mobile
- TensorFlow Lite

## 易用性工具
- FastAI
    - 高级 API：构建在 PyTorch 之上，更易用
    - 快速原型开发：快速实现前沿深度学习模型
    - 默认集成最佳实践与最新研究进展
    - 代码量少：相比原生 PyTorch，复杂模型实现更简洁
    - 优秀的迁移学习支持
- ONNX
    - 开放神经网络交换格式
    - 示例：`torch.onnx.export(model, dummy_input, "resnet18.onnx")`
    
    ```python
    import tensorflow as tf
    import tf2onnx
    import onnx
    
    # 加载 TensorFlow 模型
    tf_model = tf.keras.models.load_model('path/to/your/model.h5')
    
    # 转换为 ONNX 格式
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
    
    # 保存 ONNX 模型
    onnx.save(onnx_model, 'path/to/save/model.onnx')
    ```

    ![Untitled](assets/onnx.png)
    
- wandb
    - weights and biases 简称
    - 项目集成简单，几行代码即可
    - 团队协作方便
    - 可用直观界面对比实验结果

    ![Untitled](assets/wandb.png)
        
## 云服务商
- AWS
    - EC2 云主机
    - Sagemaker（集群上的 Jupyter Notebook、人类数据标注、模型训练和部署）
- Google Cloud
    - Vertex AI
    - 虚拟机实例
- Microsoft Azure
    - Deep speed
- OpenAI
- VastAI
    - UI 可参考相关图片
- Lambda Labs
    - 价格低廉的数据中心 GPU

## 编译器
- XLA
    - TensorFlow 专用的线性代数编译器
    - JAX 的低层优化和代码生成后端
    - 支持全程序优化，跨操作符进行整体优化
    - 自动为不同硬件（CPU、GPU、TPU）生成高效机器码
    - 实现如操作符融合等高级优化，将多个操作合并为更高效的内核
    - 让 JAX 在无需人工编写硬件特定代码的情况下获得高性能
- LLVM
- MLIR
- NVCC
    - Nvidia CUDA 编译器
    - 支持 CUDA 工具箱中的所有组件

    ![Untitled](../11_Extras/assets/nvcc.png)
        
## 其他
- Huggingface
