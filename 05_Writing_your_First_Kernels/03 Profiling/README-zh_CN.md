# 如何对你的 CUDA 内核进行性能分析

## 跟随操作步骤
1. 
```bash
nvcc -o 00 00_nvtx_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00
```

> 对于以下两个文件，你需要在 Linux 上打开 `ncu`，然后将 .nsys-rep 文件拖放到左侧边栏中。
> .sqlite 文件可以直接导入 sqlite 数据库进行更自定义的分析
2. 
```bash
nvcc -o 01 01_naive_matmul.cu`
nsys profile --stats=true ./01
```

3. 
```bash
nvcc -o 02 02_tiled_matmul.cu
nsys profile --stats=true ./02
```

## 命令行工具
- 一些用于可视化 GPU 资源使用和利用率的命令行工具
- `nvitop`
- `nvidia-smi` 或 `watch -n 0.1 nvidia-smi`


# Nsight Systems 和 Compute
- nvprof 已被弃用，因此我们将使用 `nsys` 和 `ncu` 代替
- Nsight Systems 和 Compute ⇒ `nsys profile --stats=true ./main `
![](../assets/nsight-ui.png)
- 除非你有特定的性能分析目标，否则建议的性能分析策略是从 Nsight Systems 开始，确定系统瓶颈并识别对性能影响最大的内核。[...]
- https://stackoverflow.com/questions/76291956/nsys-cli-profiling-guidance
- 如果你已经有了 `.nsys-rep` 文件，运行 `nsys stats file.nsys-rep` 可以获得更量化的分析。对于 `.sqlite` 文件，运行 `nsys analyze file.sqlite` 可以提供更定性的分析
- 要查看详细的 GUI 界面，可以运行 `nsight-sys` ⇒ 文件 ⇒ 打开 ⇒ rep 文件
- `nsys` nsight systems 是更高层次的工具；`ncu` nsight compute 是更底层的工具
- 为 Python 脚本生成性能分析文件 `nsys profile --stats=true -o mlp python mlp.py`
- 使用 nsight systems GUI 进行性能分析时，找到需要优化的内核（例如：`ampere_sgemm`），在事件视图中打开，在时间线上缩放到选定区域，通过在时间线上右键单击使用 ncu 分析内核
- ncu 可能会拒绝权限 ⇒ `code /etc/modprobe.d/nvidia.conf` 并通过添加 `options nvidia NVreg_RestrictProfilingToAdminUsers=0` 这行来强制修改 nvidia.conf，然后重启机器。详见[...]
    - 来源 ⇒ ![Nvidia 开发者论坛](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- `compute-sanitizer ./main` 用于检测内存泄漏
- 内核性能 UI ⇒ ncu-ui（可能需要运行 `sudo apt install libxcb-cursor0`）

## 内核性能分析
- [Nsight Compute 内核性能分析](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- `ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section Occupancy "./nvtx_matmul"`
- 事实证明，Nvidia 性能分析工具不会提供优化深度学习内核所需的所有信息：[详见这里](https://stackoverflow.com/questions/2204527/how-do-you-profile-optimize-cuda-kernels)

## 向量加法的性能分析
- 当对以下 3 个变体进行性能分析时，使用了一个 32（2^25）百万元素的向量加法
    - 基础版本，没有 blocks 也没有 threads
    - ![](../assets/prof1.png)
    - 使用 threads
    - ![](../assets/prof2.png)
    - 同时使用 threads 和 blocks
    - ![](../assets/prof3.png)
- 原始来源：https://developer.nvidia.com/blog/even-easier-introduction-cuda/


## NVTX 性能分析
```bash
# 编译代码
nvcc -o matmul matmul.cu -lnvToolsExt

# 使用 Nsight Systems 运行程序
nsys profile --stats=true ./matmul
```
- `nsys stats report.qdrep` 查看统计信息


## CUPTI
- 允许你构建自己的性能分析工具
- *CUDA Profiling Tools Interface*（CUPTI）使得能够创建针对 CUDA 应用程序的性能分析和追踪工具。CUPTI 提供以下 API：*Activity API*、*Callback API* [...]
- https://docs.nvidia.com/cupti/overview/overview.html
- 由于 CUPTI 的学习曲线较陡，本课程使用其他更简单的性能分析工具。