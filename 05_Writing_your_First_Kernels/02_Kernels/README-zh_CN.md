
# 内核（Kernels）

## 内核启动参数

- `dim3` 类型是用于网格（grid）和线程块（thread block）的三维类型，稍后会被传递到内核启动配置中。

- 允许以向量、矩阵或体积（张量）的方式对元素进行索引。

```cpp
dim3 gridDim(4, 4, 1); // x方向4个block，y方向4个block，z方向1个block
dim3 blockDim(4, 2, 2); // x方向4个线程，y方向2个线程，z方向2个线程
```

- 另一种类型是 `int`，用于指定一维向量

```cpp
int gridDim = 16; // 16个block
int blockDim = 32; // 每个block 32个线程
<<<gridDim, blockDim>>>
// 这些不是dim3类型，但如果索引方案是一维的也是合法的
```

- gridDim ⇒ gridDim.x * gridDim.y * gridDim.z = 启动的block总数

- blockDim ⇒ blockDim.x * blockDim.y * blockDim.z = 每个block的线程数

- 总线程数 = 每个block的线程数 × block总数

- 全局函数调用的执行配置通过插入 `<<<gridDim, blockDim, Ns, S>>>` 这样的表达式来指定，其中：

	- Dg（dim3）指定网格的维度和大小。
	- Db（dim3）指定每个block的维度和大小。
	- Ns（size_t）指定为本次调用每个block动态分配的共享内存字节数，除了静态分配的内存外（通常省略）。
	- S（cudaStream_t）指定关联的流，是一个可选参数，默认为0。

> 来源 -> https://stackoverflow.com/questions/26770123/understanding-this-cuda-kernels-launch-parameters

## 线程同步

- `cudaDeviceSynchronize();` ⇒ 确保一个问题的所有内核都已完成执行，这样你就可以安全地开始下一个任务。可以将其视为一个屏障（barrier）。此函数应在 `int main()` 或其他**非** `__global__` 函数中调用。

- `__syncthreads();` 用于在内核内部设置线程执行的屏障。如果你在操作同一块内存，并且需要所有其他线程都赶上进度后再对某个位置进行修改时很有用。例如：某个线程可能还在处理一块内存，另一个线程已经完成了该任务。如果这个更快的线程修改了慢线程还需要的数据，就会导致数值不稳定和错误。

- `__syncwarps();` 同步一个warp内的所有线程。

- 为什么需要同步线程？因为线程是异步的，可以以任意顺序执行。如果一个线程依赖于另一个线程的结果，就需要确保被依赖的线程先完成。

- 例如，如果我们要对两个数组 `a = [1, 2, 3, 4]`，`b = [5, 6, 7, 8]` 做向量加法并将结果存入 `c`，然后再对 `c` 的每个元素加1，我们需要确保所有乘法操作完成后再进行加法（遵循运算优先级）。如果此处不进行线程同步，可能会出现错误的输出结果，比如1在乘法之前就被加上了。

- 更清晰但不常见的例子是并行位移操作。如果某个位移操作依赖于上一个位移操作的结果，就需要确保前一个操作完成后再进行下一个。
	![](../assets/bitshift1.png)

![](../assets/barrier.png)

## 线程安全

### [CUDA 是线程安全的吗？](https://forums.developer.nvidia.com/t/is-cuda-thread-safe/2262/2)

- 当一段代码是“线程安全”的，意味着它可以被多个线程同时运行，而不会导致竞态条件或其他意外行为。

- 竞态条件是指一个线程在另一个线程完成之前就开始了下一个任务。为了防止竞态条件，我们使用 `cudaDeviceSynchronize()` 这样的特殊函数，确保所有线程都赶上进度后再给它们新的指令。可以想象一群线程在赛跑，有些线程先到终点，你需要手动让这些“赢家”线程在终点等慢的线程。

- 如果你想了解如何用不同的CPU线程调用多个GPU内核，请参考上面的链接。

## SIMD/SIMT（单指令多线程）

### [CUDA 可以使用SIMD指令吗？](https://stackoverflow.com/questions/5238743/can-cuda-use-simd-extensions)
- 类似于CPU的SIMD（单指令多数据），GPU上有单指令多线程（SIMT）。
- 与其让for循环顺序执行，不如让每个线程执行for循环的一次迭代，这样看起来只需要一次迭代的时间。如果迭代次数增加，执行时间会线性增长（因为并行核心数量有限）。
- 比CPU更简单：
	- 顺序发射指令
	- 没有分支预测
	- 控制逻辑远少于CPU架构，因此可以有更多的核心

> 本课程后面（矩阵乘法优化章节）会回到与这些特殊warp操作相关的优化。[Warp级原语](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy 指出：“每个block的线程数是有限制的，因为所有block内的线程都要驻留在同一个流多处理器（SM）上，并且必须共享该核心有限的内存资源。**在当前GPU上，一个线程块最多可以包含1024个线程。” 这意味着每个block理论上最多1024个线程，每个warp 32个线程，每个block 32个warp。**

## 数学内建函数（Math intrinsics）
- 设备端专用的硬件数学指令
- https://docs.nvidia.com/cuda/cuda-math-api/index.html
- 你可以使用主机端的操作如 `log()`（主机），而不是 `logf()`（设备），但主机端会更慢。这些数学基础函数允许在设备/GPU上高效执行数学运算。你可以在nvcc编译器中传递 `-use_fast_math` 参数，将其转换为这些设备专用操作，代价是精度损失极小。
- `--fmad=true` 用于启用融合乘加（fused multiply-add）
