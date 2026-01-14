# CUDA 流（Streams）示例

## 直观理解
可以把 stream 理解为“河道里的水流”，操作只沿着时间轴向前推进（就像一条时间线）。例如，先把一些数据拷过去（时间步 1），再进行一段计算（时间步 2），然后继续后续的操作（时间步 3），依次推进。

在 CUDA 中可以同时拥有多个 stream，并且每个 stream 都有自己独立的时间线。这使我们可以重叠不同操作，更高效地利用 GPU。

在训练超大语言模型时，把大量时间浪费在把所有 token 在 GPU 与主机之间来回搬运是非常不划算的。借助 streams，我们可以在进行计算的同时搬运数据，从而持续保持 GPU 的忙碌度和吞吐。

本项目演示了如何使用 CUDA streams 来实现并发执行并更好地利用 GPU。它包含两个示例：

## 代码片段
- 默认的 stream = stream 0 = null stream（空流）
```cpp
// 该 kernel 的启动使用了空流（0）
myKernel<<<gridSize, blockSize>>>(args);

// 等价于
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

还记得在 Kernels 部分的这段吗？
- 全局函数调用（kernel 启动）的执行配置通过形如 `<<<gridDim, blockDim, Ns, S>>>` 的表达式来指定，其中：

  - Dg（dim3）指定网格（grid）的维度与大小。
  - Db（dim3）指定每个线程块（block）的维度与大小。
  - Ns（size_t）指定为本次调用、按块动态分配的共享内存字节数（在静态分配之外；通常省略）。
  - S（cudaStream_t）指定关联的 stream，是一个可选参数，默认值为 0。

- stream1 与 stream2 使用不同优先级创建。这意味着它们在运行时会以一定顺序被调度执行，从而让我们对内核（kernel）的并发执行拥有更精细的控制。

```cpp
// 以不同优先级创建流
int leastPriority, greatestPriority;
CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

## 示例

1. `stream_basics.cu`：演示使用异步内存传输与内核启动的基础 stream 用法。
2. `stream_advanced.cu`：演示更高级的话题，例如 stream 优先级、回调（callback）以及跨 stream 的依赖关系。

## 编译

使用以下命令编译示例：

```bash
nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu
```

## 文档
- [Streams and Concurrency Webinar (NVIDIA)](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

## 锁页内存（Pinned Memory）
- 可以把它理解为：“这块我们后面还要用，别动它。”
- 锁页内存是被固定在物理内存中的页，操作系统不会在后台移动它。这在你想把数据传到 GPU 并在其上进行计算时非常有用；固定后可让 DMA 传输稳定高效，避免页迁移带来的性能问题或潜在中断。

```cpp
// 分配锁页内存
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

## 事件（Events）
- 测量内核执行时间：在内核启动前后放置事件，可以准确测量执行时间。
- 在不同 stream 之间同步：事件可用于在不同 stream 之间建立依赖，确保一个操作只在另一个完成后才开始。
- 计算与数据传输重叠：事件可以标记一次数据传输的完成，以便在数据就绪后开始计算。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## 回调（Callbacks）
- 通过回调，你可以搭建一条流水线：当 GPU 上的某个操作完成时，会触发 CPU 端的回调以启动后续工作，CPU 随后可以继续向 GPU 提交更多任务，从而形成持续的异步处理流程。

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // 触发下一批任务
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```