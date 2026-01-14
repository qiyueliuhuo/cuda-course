# 什么是原子操作（Atomic Operations）
    
这里的“atomic（原子）”指的是物理学中的“不可再分”的概念，即某个事物不能再被进一步分解。

“原子操作”（atomic operation）保证对某个内存位置的一次特定操作会由一个线程完整完成，在此之前，其他线程无法访问或修改同一内存位置。这样可以防止并发访问导致的数据竞争（race condition）和不一致问题。

由于在原子操作期间限制了单位时间内对同一内存位置的操作量，性能会有一定损失。它由硬件保证内存访问的安全性，但以一定的速度为代价。

### 整数原子操作（Integer Atomic Operations）

- `atomicAdd(int* address, int val)`：以原子方式将 `val` 加到 `address` 指向的值上，并返回旧值。
- `atomicSub(int* address, int val)`：以原子方式从 `address` 指向的值中减去 `val`，并返回旧值。
- `atomicExch(int* address, int val)`：以原子方式将 `address` 指向的值与 `val` 交换，并返回旧值。
- `atomicMax(int* address, int val)`：以原子方式将 `address` 指向的值设置为当前值与 `val` 的较大者。
- `atomicMin(int* address, int val)`：以原子方式将 `address` 指向的值设置为当前值与 `val` 的较小者。
- `atomicAnd(int* address, int val)`：以原子方式对 `address` 指向的值与 `val` 执行按位与（AND）。
- `atomicOr(int* address, int val)`：以原子方式对 `address` 指向的值与 `val` 执行按位或（OR）。
- `atomicXor(int* address, int val)`：以原子方式对 `address` 指向的值与 `val` 执行按位异或（XOR）。
- `atomicCAS(int* address, int compare, int val)`：以原子方式比较 `address` 指向的值与 `compare`，若相等则将其替换为 `val`，并返回原值。

### 浮点原子操作（Floating-Point Atomic Operations）

- `atomicAdd(float* address, float val)`：以原子方式将 `val` 加到 `address` 指向的值上，并返回旧值。自 CUDA 2.0 起可用。
- 注意：对双精度变量的浮点原子操作自 CUDA 计算能力（Compute Capability）6.0 起支持，可使用 `atomicAdd(double* address, double val)`。

### 从零实现（From Scratch）

现代 GPU 具备专用硬件指令以高效完成这些操作，底层通常使用类似 CAS（Compare-and-Swap，比较并交换）之类的技术。

你可以把原子操作看作一种非常快速的硬件级互斥（mutex）。就好像每次原子操作会执行如下步骤：

1. lock(memory_location)
2. old_value = *memory_location
3. *memory_location = old_value + increment
4. unlock(memory_location)
5. return old_value

```cpp
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    while (atomicCAS(&lock, 0, 1) != 0);  // 获取锁（自旋）
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // 确保写入对其他线程可见
    
    atomicExch(&lock, 0);  // 释放锁
    
    return old;
}
```

- 互斥（Mutual Exclusion） ⇒ [Mutual Exclusion（视频）](https://www.youtube.com/watch?v=MqnpIwN7dz0&t)
- “Mutual（相互的）”：
  - 暗示实体之间（此处为线程或进程）的一种互相/共享关系。
  - 表示该排他性对所有参与方等同适用。
- “Exclusion（排除）”：
  - 指将某事物排除在外或阻止其访问的行为。
  - 在这里，指阻止对某一资源的并发访问。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 我们的互斥量结构
struct Mutex {
    int *lock;
};

// 初始化互斥量
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// 获取互斥量（加锁）
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // 自旋等待
    }
}

// 释放互斥量（解锁）
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// 演示互斥量使用的内核函数
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // 临界区
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动包含多个线程的内核
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}
```