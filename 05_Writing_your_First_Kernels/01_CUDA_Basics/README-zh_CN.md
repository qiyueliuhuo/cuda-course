# CUDA 基础

## 打印你的 GPU 的一些统计信息
![](../assets/gpustats.png)

## 基础知识
主机（Host） ⇒ CPU ⇒ 使用主板上的内存（RAM）

设备（Device） ⇒ GPU ⇒ 使用芯片上的显存（VRAM，视频内存）

CUDA 程序的表层运行流程：

1. 将输入从主机（Host）复制到设备（Device）
2. 加载 GPU 程序并使用传输到设备上的数据执行
3. 将结果从设备（Device）复制回主机（Host），以便显示或进一步处理

## Device VS Host 命名规范
`h_A` 表示主机（CPU）上的变量“A”

`d_A` 表示设备（GPU）上的变量“A”

`__global__` 声明为全局可见，表示主机（CPU）可以调用这些全局函数。这些函数通常不返回值，而是对传入的变量执行快速操作。

`__device__` 声明的函数仅供 GPU 调用。例如，一个原始的注意力分数矩阵只存在于 GPU 上，只有设备端可以直接操作它。

`__host__` 仅在 CPU 上运行。与普通的 C/C++ 脚本在没有 CUDA 时的运行方式相同。

## 内存管理

- `cudaMalloc` 用于在显存（VRAM，也称全局内存）上分配内存

```cpp
float *d_a, *d_b, *d_c;

cudaMalloc(&d_a, N*N*sizeof(float));
cudaMalloc(&d_b, N*N*sizeof(float));
cudaMalloc(&d_c, N*N*sizeof(float));
```

- `cudaMemcpy` 用于在设备与主机之间或设备内部进行数据复制
    - 主机到设备 ⇒ CPU 到 GPU
    - 设备到主机 ⇒ GPU 到 CPU
    - 设备到设备 ⇒ GPU 上的不同位置之间
    - 具体方向用 **`cudaMemcpyHostToDevice`**、**`cudaMemcpyDeviceToHost`** 或 **`cudaMemcpyDeviceToDevice`** 指定
- `cudaFree` 用于释放设备上的内存

# `nvcc` 编译器
- 主机代码
    - 修改后运行内核
    - 编译为 x86 二进制

- 设备代码
    - 编译为 PTX（并行线程执行）
    - 可在不同 GPU 架构间稳定运行

- JIT（即时编译）
    - 支持向后兼容

## CUDA 层级结构
1. 内核在单个线程中执行
2. 线程被分为线程块（Thread Block，也称为 Block）
3. 块被组合成一个网格（Grid）
4. 内核以一个包含多个块和线程的网格形式执行

### 四个技术术语：
- `gridDim` ⇒ 网格中的块数
- `blockIdx` ⇒ 块在网格中的索引
- `blockDim` ⇒ 一个块中的线程数
- `threadIdx` ⇒ 线程在块中的索引

（更多内容见视频讲解）

## 线程（Threads）
- 每个线程有自己的局部内存（寄存器），线程私有
- 例如要计算 `a = [1, 2, 3, ... N]` 和 `b = [2, 4, 6, ... N]` 的加法，每个线程只做一次加法 ⇒ `a[0] + b[0]`（线程1）；`a[1] + b[1]`（线程2）；以此类推

## 线程束（Warp）

> Warp 是 NVIDIA GPU 上的一个硬件调度单位，每个 Warp 通常包含 32 个并行执行的线程。Warp 内的线程会同步执行同一条指令（SIMT 架构），因此翻译为“线程束”最符合技术含义，并且已被中文技术社区广泛接受。

![](../assets/weft.png)
- https://en.wikipedia.org/wiki/Warp_and_weft
- Warp 指的是在织布机上先被拉紧的纱线集合，Weft 则是在织造过程中穿插进去的部分
- 每个 Warp在一个 Block 内部，并行处理 32 个线程
- 指令下发到 Warp，由 Warp 调度线程执行（而不是直接下发给线程）
- Warp 是不可避免的 CUDA 执行单位
- Warp 调度器负责调度 Warp 运行
- 每个 SM有 4 个 Warp 调度器
![](../assets/schedulers.png)

## 块（Blocks）
- 每个块有共享内存（可被同一个线程块内所有线程访问）
- 在不同数据上执行相同的代码，共享内存空间，使得协同读写更高效

## 网格（Grids）
- 内核执行时，网格中的块和块内的线程可以访问全局内存（VRAM）
- 网格包含多个块，最典型的应用是批处理，每个块对应一个批次元素

> 为什么不用纯线程而要有块和线程的分层？结合我们对 Warp 如何以锁步（lockstep）方式执行 32 个线程的理解，可以更好地掌握批量操作和资源分配
> 从逻辑上讲，块间的共享内存是独立分区的。也就是说，线程可以通过块的共享内存与同块内其他线程通信。

- CUDA 的并行性可扩展，因为块的运行没有严格的顺序依赖。例如，不一定要先运行 Block 0 & Block 1，再运行 Block 2 & 3……也可能先运行 Block 3 & 0，再运行 Block 1 & 2。

> [线程如何映射到 CUDA 核心？](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)