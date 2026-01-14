## 操作中的关键细节（Nuts and Bolts of the operations）

下面是一些 CUDA 的基础概念和细节，在后续会被无数次用到。

### Kernel（核函数）

Kernel 是一种在 **GPU（显卡）** 上运行的特殊函数，而不是在 CPU 上运行。  
可以把它想象成你给一大群工人（GPU 线程）下达的指令，他们可以**同时并行工作**。  
使用 `__global__` 关键字标记 kernel，它的返回类型只能是 `void`。示例：

```cpp
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

### Grid（网格）

Grid 表示一次 kernel 启动时创建的 **全部线程集合**，可以理解为整体的执行空间。  
它是由许多线程块（block）组成的集合。

当你启动一个 kernel 时，你需要指定 grid 的维度，从而定义会创建多少个 block。  
Grid 可以是一维、二维或三维（像一条线、一张平面或一个立方体的 block 布局）。  
它主要用于组织**非常大的计算任务**。

- 示例：处理一张大图像时，可以让 **每个 block 负责图像的一个分块区域**。

### Block（线程块）

Block 是一组线程的集合，这些线程可以协作，并通过 **共享内存（shared memory）** 进行快速数据共享。  
Block 同样可以是一维、二维或三维。

- 一个 block 内的线程可以：
  - 共享内存
  - 彼此同步
  - 协同完成任务

- 示例：在处理图像时，一个 block 可以负责一个 `16x16` 像素的小区域。

### Threads（线程）

线程是 CUDA 中最小的执行单元。每个线程会 **独立地执行 kernel 中的代码**。  
在一个 block 中，线程通过唯一的线程 ID 进行标识。  
这个 ID 允许你访问特定的数据，或者根据线程在 block 中的位置，执行不同的操作。

> 每个线程都有自己唯一的 ID，用来确定它应该处理哪一部分数据。

---

### 理解 CUDA 线程索引（CUDA Thread Indexing）

在 CUDA 中，每个线程都有一个唯一的标识符，可以用来确定它在 **grid 与 block 中的位置**。  
常用的内置变量如下：

1. **`threadIdx`**：  
   - 一个三维向量（`threadIdx.x`, `threadIdx.y`, `threadIdx.z`），表示线程在其 **所属 block 内** 的位置。
   - 示例：如果你有一个一维 block，包含 256 个线程，则 `threadIdx.x` 的范围是 `0` 到 `255`。

2. **`blockDim`**：  
   - 一个三维向量（`blockDim.x`, `blockDim.y`, `blockDim.z`），指定 **block 的线程维度大小**。
   - 示例：如果你的 block 在 x 方向上有 256 个线程，则 `blockDim.x = 256`。

3. **`blockIdx`**：  
   - 一个三维向量（`blockIdx.x`, `blockIdx.y`, `blockIdx.z`），表示当前 block 在 **整个 grid 中的位置**。
   - 示例：如果你有一个一维 grid，包含 10 个 block，则 `blockIdx.x` 的范围是 `0` 到 `9`。

4. **`gridDim`**：  
   - 一个三维向量（`gridDim.x`, `gridDim.y`, `gridDim.z`），指定 **grid 的 block 维度大小**。
   - 示例：如果你的 grid 在 x 方向上有 10 个 block，则 `gridDim.x = 10`。

---

### 计算全局线程 ID（Global Thread ID）

当需要为一维数组中的每个元素分配一个线程时，可以通过下面的公式计算线程在整个 grid 中的 **全局唯一 ID**：

```cpp
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x * blockDim.x` 给出了当前 block 在整个 grid 中的起始索引（之前所有 block 里的线程总数）。
- `threadIdx.x` 给出了当前线程在本 block 内的偏移。
- 两者相加就是这个线程在所有线程中的全局 ID。

---

## 辅助类型和函数（Helper Types and Functions）

#### `dim3`

- 一个简单的类型，用来指定三维尺寸
- 通常用于配置 grid 和 block 的大小
- 示例：

```cpp
dim3 blockSize(16, 16, 1);  // 每个 block 有 16x16x1 个线程
dim3 gridSize(8, 8, 1);     // grid 中有 8x8x1 个 block
```

#### `<<< >>>`

这个语法用于在 GPU 上 **配置并启动 kernel**。它可以指定：

- grid 维度
- block 维度
- 共享内存大小（可选）
- 执行流（stream， 可选）

合理地配置这些参数，对于高效地使用 GPU 至关重要。

- 用于启动 kernel 的特殊尖括号语法
- 用于指定 grid 和 block 的维度
- 示例：

```cpp
addNumbers<<<gridSize, blockSize>>>(a, b, result);
```

其中 `addNumbers` 是 kernel 的名字，`gridSize` 和 `blockSize` 分别是 grid 和 block 的大小，`a`、`b`、`result` 是传入 kernel 的参数。

---

## 内存管理（Memory Management）

### `cudaMalloc`

- 在 GPU 上分配内存
- 类似于普通的 `malloc`，但作用于 **设备（device）内存**
- 示例：

```cpp
int *device_array;
cudaMalloc(&device_array, size * sizeof(int));
```

### `cudaMemcpy`

用于在 **主机（host，CPU）与设备（device，GPU）之间** 或 **设备内部** 拷贝数据：

- host to device ⇒ 从 CPU 拷贝到 GPU
- device to host ⇒ 从 GPU 拷贝到 CPU
- device to device ⇒ 在 GPU 不同位置之间拷贝

常用拷贝方向常量：

- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`

- `cudaFree` 用于释放在设备上分配的内存。

示例：

```cpp
cudaMemcpy(device_array, host_array, size * sizeof(int), cudaMemcpyHostToDevice);
```

### `cudaDeviceSynchronize()`

默认情况下，CPU 和 GPU 会 **并行工作**，以尽量减少总执行时间。  
但有时，我们需要让 CPU **等待所有 GPU 操作完成**，这时就要使用 `cudaDeviceSynchronize()`。

- 当你需要在继续执行 CPU 端代码前，确保 GPU 结果已经就绪时，这个函数就很有用。
- 在做性能基准测试时，我们也会用它来确保测量的时间包括完整的 kernel 执行。

示例：

```cpp
kernel<<<gridSize, blockSize>>>(data);
cudaDeviceSynchronize();  // 等待 kernel 执行完毕
printf("Kernel completed!");
```

---

## 一些关于内存的理论（Some theory on Memory）

CUDA 提供了多种类型的内存，每种都有不同的访问速度和适用场景：

1. **Global Memory（全局内存）**  
   - GPU 的主存，所有线程都可以访问。  
   - **最慢**，但容量 **最大**。  
   - 适合在所有线程之间共享的大规模数据。  
   - 示例：数组或大型数据集。

2. **Shared Memory（共享内存）**  
   - 一个 block 内所有线程共享的内存。  
   - **非常快**，但容量 **较小**。  
   - 适合 block 内线程频繁共享和重复访问的数据。  
   - 示例：临时变量、小型查找表。

3. **Registers（寄存器）**  
   - 访问速度最快，每个线程 **私有**。  
   - 一般用于存储局部变量。  
   - 数量有限，要谨慎使用。  
   - 示例：循环计数器、中间计算结果。

4. **Constant Memory（常量内存）**  
   - 所有线程只读的内存。  
   - 有缓存机制，访问速度较快。  
   - 适合在 kernel 执行期间 **不会改变** 的数据。  
   - 示例：常量、配置参数等。

5. **Local Memory（本地内存）**  
   - 当寄存器不够用时，一些数据会溢出到这里。  
   - 实际上存放在全局内存中，因此**访问较慢**。  
   - 应尽可能避免使用（意味着寄存器被用爆了）。  
   - 示例：过大的局部数组，或放不进寄存器的本地变量。

---

## 综合示例（Putting It All Together）

下面是一个简单的示例，把上述概念串联起来：

```cpp
// Kernel 定义
__global__ void addArrays(int *a, int *b, int *c, int size) {
    // 计算当前线程的全局唯一索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保不越界
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 在 main 函数中：
dim3 blockSize(256);  // 每个 block 256 个线程
dim3 gridSize((size + blockSize.x - 1) / blockSize.x);  // 计算需要多少个 block
addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
```

这个例子展示了线程是如何组织起来完成两个数组的并行相加的：

- 每个线程负责处理数组中的一个元素。
- 我们通过 block 和 grid 的维度设置，确保整个数组都被覆盖。
- 通过 `index < size` 的判断，避免线程访问越界。