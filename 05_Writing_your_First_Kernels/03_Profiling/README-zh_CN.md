# 如何对你的 CUDA 内核进行性能分析

## 跟随操作步骤
1. 
```bash
nvcc -o 00 00_nvtx_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00
```

```bash
> sudo nsys profile --stats=true ./00
Collecting data...  # 开始采集性能数据
Generating '/tmp/nsys-report-6b75.qdstrm'  # 生成中间数据流文件
[1/8] [========================100%] report1.nsys-rep  # 生成 .nsys-rep 分析报告 (供GUI查看)
[2/8] [========================100%] report1.sqlite  # 生成 SQLite 数据库 (供脚本查询)
[3/8] Executing 'nvtx_sum' stats report  # NVTX 手动标记区域统计 (逻辑层耗时)

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)   Style           Range         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  -------  ----------------------
     50.0      329,694,056          1  329,694,056.0  329,694,056.0  329,694,056  329,694,056          0.0  PushPop  :Matrix Multiplication  # 业务逻辑总耗时 (含内存分配、拷贝及计算)
     49.0      323,408,432          1  323,408,432.0  323,408,432.0  323,408,432  323,408,432          0.0  PushPop  :Memory Allocation     # 显存分配逻辑耗时
      0.4        2,809,940          1    2,809,940.0    2,809,940.0    2,809,940    2,809,940          0.0  PushPop  :Kernel Execution      # Kernel 逻辑耗时
      0.3        1,732,465          1    1,732,465.0    1,732,465.0    1,732,465    1,732,465          0.0  PushPop  :Memory Copy D2H       # D2H 拷贝逻辑耗时
      0.2        1,428,606          1    1,428,606.0    1,428,606.0    1,428,606    1,428,606          0.0  PushPop  :Memory Copy H2D       # H2D 拷贝逻辑耗时
      0.0          311,175          1      311,175.0      311,175.0      311,175      311,175          0.0  PushPop  :Memory Deallocation   # 显存释放耗时 (可忽略)

[4/8] Executing 'osrt_sum' stats report  # 操作系统运行时统计 (CPU 系统调用)

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     71.6      286,286,273         11  26,026,024.8  2,252,516.0   133,774  211,857,494  62,610,804.5  poll                  # CPU 轮询驱动文件句柄 (等待 GPU 完成时的阻塞开销)
     27.6      110,179,271        471     233,926.3      8,853.0     1,191   12,327,993     816,219.7  ioctl                 # 设备驱动控制指令 (频繁的用户态/内核态切换)
      0.3        1,346,384         25      53,855.4      7,536.0     6,047      899,882     177,692.4  mmap64                # 显存/内存映射操作 (通常伴随 cudaMalloc)
      0.2          739,468          9      82,163.1     57,619.0    30,563      269,033      72,014.1  sem_timedwait         # 信号量等待 (线程同步)
      0.1          224,276         43       5,215.7      4,388.0     2,695       12,350       2,160.9  open64                # 打开文件/设备
      0.0          170,157          3      56,719.0     59,483.0    36,350       74,324      19,137.3  pthread_create        # 线程创建开销
      0.0          164,576          1     164,576.0    164,576.0   164,576      164,576           0.0  pthread_cond_wait     # 条件变量等待
      0.0          128,289         27       4,751.4      3,283.0     1,138       16,082       3,796.6  fopen                 # 打开文件
      0.0          111,976         13       8,613.5      5,149.0     1,692       49,634      12,701.8  mmap                  # 内存映射
      0.0           32,877         19       1,730.4      1,402.0     1,007        4,540         963.5  fclose                # 关闭文件
      0.0           31,540          8       3,942.5      2,882.0     1,058        8,736       3,099.8  close                 # 关闭文件描述符
      0.0           28,788          1      28,788.0     28,788.0    28,788       28,788           0.0  fgets                 # 字符串读取
      0.0           23,027          6       3,837.8      3,953.5     1,120        6,738       1,856.6  open                  # 打开文件
      0.0           22,685          5       4,537.0      2,952.0     1,423       10,423       3,585.0  fread                 # 二进制读取
      0.0           15,528          3       5,176.0      4,474.0     3,961        7,093       1,679.9  munmap                # 解除内存映射
      0.0           12,861          9       1,429.0      1,056.0     1,010        2,790         660.0  read                  # 读取文件
      0.0           10,534          7       1,504.9      1,360.0     1,054        2,131         445.6  write                 # 写入文件
      0.0           10,138          3       3,379.3      4,137.0     1,155        4,846       1,958.7  pipe2                 # 创建管道
      0.0            9,420          2       4,710.0      4,710.0     3,956        5,464       1,066.3  socket                # 创建套接字
      0.0            7,175          1       7,175.0      7,175.0     7,175        7,175           0.0  connect               # 连接套接字
      0.0            2,731          1       2,731.0      2,731.0     2,731        2,731           0.0  fcntl                 # 文件控制
      0.0            1,879          1       1,879.0      1,879.0     1,879        1,879           0.0  fwrite                # 写入文件
      0.0            1,608          1       1,608.0      1,608.0     1,608        1,608           0.0  bind                  # 绑定地址
      0.0            1,290          1       1,290.0      1,290.0     1,290        1,290           0.0  pthread_cond_broadcast # 广播条件变量
      0.0            1,248          1       1,248.0      1,248.0     1,248        1,248           0.0  fflush                # 刷新缓冲区

[5/8] Executing 'cuda_api_sum' stats report  # CUDA Runtime API 统计 (Host 端开销)

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  ---------  ----------  ------------  ----------------------
     93.8       94,994,088          3  31,664,696.0     49,552.0     48,360  94,896,176  54,760,068.0  cudaMalloc            # 显存分配 API (首次调用包含 Context 初始化，耗时巨大)
      3.1        3,154,424          3   1,051,474.7    744,414.0    678,827   1,731,183     589,557.4  cudaMemcpy            # 数据拷贝 API (Host 端发起传输的调用开销)
      2.6        2,645,612          1   2,645,612.0  2,645,612.0  2,645,612   2,645,612           0.0  cudaDeviceSynchronize # 显式同步 API (CPU 阻塞等待 GPU)
      0.3          309,163          3     103,054.3    119,163.0     65,952     124,048      32,224.3  cudaFree              # 显存释放 API
      0.2          161,235          1     161,235.0    161,235.0    161,235     161,235           0.0  cudaLaunchKernel      # Kernel 启动 API (将指令放入 Command Buffer，不含执行)
      0.0              677          1         677.0        677.0        677         677           0.0  cuModuleGetLoadingMode # 模块加载状态查询

[6/8] Executing 'cuda_gpu_kern_sum' stats report  # GPU Kernel 执行统计 (Device 端真实耗时)

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                       Name                      
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -----------------------------------------------
    100.0        2,644,050          1  2,644,050.0  2,644,050.0  2,644,050  2,644,050          0.0  matrixMulKernel(...)    # 矩阵乘法 Kernel 实际运行耗时 (2.6ms)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report  # GPU 内存传输时间统计 (硬件传输时间)

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Operation          
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ----------------------------
     53.4        1,154,050      2    577,025.0    577,025.0    576,465    577,585        792.0  [CUDA memcpy Host-to-Device]  # H2D 数据上行耗时 (总计 1.15ms)
     46.6        1,008,390      1  1,008,390.0  1,008,390.0  1,008,390  1,008,390          0.0  [CUDA memcpy Device-to-Host]  # D2H 数据回传耗时 (总计 1.00ms)

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report  # GPU 内存传输量统计 (吞吐量分析)

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]  # H2D 传输总量 (~8.4MB)
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]  # D2H 传输总量 (~4.2MB)

Generated:
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.nsys-rep
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite
```

```bash
> nsys analyze report1.sqlite 
Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/cuda_memcpy_async.py]... 

 ** CUDA Async Memcpy with Pageable Memory (cuda_memcpy_async):

There were no problems detected related to memcpy operations using pageable
memory.  # 未检测到使用分页内存（Pageable Memory）进行异步拷贝的低效行为

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/cuda_memcpy_sync.py]... 

 ** CUDA Synchronous Memcpy (cuda_memcpy_sync):

The following are synchronous memory transfers that block the host. This does
not include host to device transfers of a memory block of 64 KB or less.
# 下表列出了阻塞 Host (CPU) 的同步内存传输（忽略小于64KB的小包传输）

Suggestion: Use cudaMemcpy*Async() APIs instead.  # 建议：改用 cudaMemcpyAsync 实现 CPU/GPU 重叠执行

 Duration (ns)  Start (ns)    Src Kind  Dst Kind  Bytes (MB)     PID      Device ID  Context ID  Green Context ID  Stream ID      API Name     
 -------------  -----------  --------  --------  ----------  ---------  ---------  ----------  ----------------  ---------  ----------------
     1,008,390  360,484,139  Device    Pageable       4.194  1,373,105          0           1                 7  cudaMemcpy_v3020  # D2H (设备到主机) 拷贝，阻塞 CPU 约 1ms
       577,585  357,164,672  Pageable  Device         4.194  1,373,105          0           1                 7  cudaMemcpy_v3020  # H2D (主机到设备) 拷贝，阻塞 CPU 约 0.58ms
       576,465  356,419,977  Pageable  Device         4.194  1,373,105          0           1                 7  cudaMemcpy_v3020  # H2D 拷贝，阻塞 CPU 约 0.58ms (注意 Src 是分页内存)

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/cuda_memset_sync.py]... 
SKIPPED: report1.sqlite could not be analyzed because it does not contain the required CUDA data. Does the application use CUDA memset APIs?
# 跳过检查：日志中无 memset 数据（说明程序未使用 cudaMemset）

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/cuda_api_sync.py]... 

 ** CUDA Synchronization APIs (cuda_api_sync):

The following are synchronization APIs that block the host until all issued
CUDA calls are complete.
# 下表列出了会完全阻塞 Host 直到 GPU 任务完成的同步 API

Suggestions:
   1. Avoid excessive use of synchronization.  # 建议1：避免过度同步（会破坏流水线并行）
   2. Use asynchronous CUDA event calls, such as cudaStreamWaitEvent() and
cudaEventSynchronize(), to prevent host synchronization.  # 建议2：使用 Event 进行细粒度同步

 Duration (ns)  Start (ns)       PID       TID           API Name           
 -------------  -----------  ---------  ---------  ---------------------------
     2,645,612  357,826,729  1,373,105  1,373,105  cudaDeviceSynchronize_v3020  # 全局同步点，阻塞 CPU 约 2.6ms (等待 Kernel 执行完成)

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/gpu_gaps.py]... 

 ** GPU Gaps (gpu_gaps):

There were no problems detected with GPU utilization. GPU was not found to be
idle for more than 500ms.  # 未检测到 GPU 长时间空闲（超过500ms的空档）

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/gpu_time_util.py]... 

 ** GPU Time Utilization (gpu_time_util):

The following are time regions with an average GPU utilization below 50%%.
Addressing the gaps might improve application performance.
# 下表列出了 GPU 利用率低于 50% 的时间段（流水线气泡）

Suggestions:
   1. Use CPU sampling data, OS Runtime blocked state backtraces, and/or OS
Runtime APIs related to thread synchronization to understand if a sluggish or
blocked CPU is causing the gaps.  # 建议1：检查 CPU 是否因阻塞导致未能及时向 GPU 提交任务
   2. Add NVTX annotations to CPU code to understand the reason behind the gaps.  # 建议2：添加 NVTX 标记以分析 CPU 逻辑

 Row#  In-Use (%)  Duration (ns)  Start (ns)       PID      Device ID  Context ID
 ----  ----------  -------------  -----------  ---------  ---------  ----------
    1        40.9        169,086  356,927,235  1,373,105          0           1  # 在这 0.17ms 区间内，GPU 仅有 40.9% 的时间在工作 (可能是 H2D 与 Kernel 之间的调度间隙)

Processing [report1.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/rules/dx12_mem_ops.py]... 
SKIPPED: report1.sqlite could not be analyzed because it does not contain the required DX12 data. Does the application use DX12 APIs?
# 跳过检查：非 Windows DirectX12 应用
```

> 对于以下两个文件，你需要在 Linux 上打开 `ncu`，然后将 .nsys-rep 文件拖放到左侧边栏中。
> .sqlite 文件可以直接导入 sqlite 数据库进行更自定义的分析
2. 
```bash
nvcc -o 01 01_naive_matmul.cu
nsys profile --stats=true ./01
```

```bash
> sudo nsys profile --stats=true ./01
Collecting data...
Generating '/tmp/nsys-report-a7b0.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     73.3      300,103,279         11  27,282,116.3  9,914,856.0   269,765  199,732,574  58,392,509.4  poll                  
     26.0      106,442,210        460     231,396.1      8,925.0     1,098   11,849,069     777,808.1  ioctl                 
      0.3        1,378,097         25      55,123.9      7,415.0     6,802      906,084     178,917.9  mmap64                
      0.2          739,545          9      82,171.7     58,944.0    30,016      273,553      72,914.5  sem_timedwait         
      0.1          239,305         43       5,565.2      4,552.0     2,571       23,426       3,272.5  open64                
      0.0          163,927         29       5,652.7      3,634.0     1,179       16,495       4,474.2  fopen                 
      0.0          126,490          3      42,163.3     44,179.0    32,668       49,643       8,665.2  pthread_create        
      0.0          125,030         13       9,617.7      5,389.0     1,886       50,701      13,032.0  mmap                  
      0.0           84,413          1      84,413.0     84,413.0    84,413       84,413           0.0  pthread_cond_wait     
      0.0           37,121         19       1,953.7      1,592.0     1,088        4,584         946.1  fclose                
      0.0           36,353          1      36,353.0     36,353.0    36,353       36,353           0.0  fgets                 
      0.0           35,114          9       3,901.6      2,968.0     1,022        9,751       3,285.7  close                 
      0.0           23,118          6       3,853.0      3,760.0     1,217        6,696       1,806.4  open                  
      0.0           22,980          4       5,745.0      4,754.0     2,850       10,622       3,589.3  fread                 
      0.0           14,668          3       4,889.3      1,910.0     1,240       11,518       5,750.4  fwrite                
      0.0           14,642         11       1,331.1      1,110.0     1,011        2,377         443.6  read                  
      0.0           12,486          2       6,243.0      6,243.0     4,590        7,896       2,337.7  socket                
      0.0           11,215          8       1,401.9      1,359.5     1,052        2,064         353.8  write                 
      0.0            9,819          3       3,273.0      4,089.0     1,518        4,212       1,521.1  pipe2                 
      0.0            9,532          3       3,177.3      3,118.0     2,943        3,471         269.0  munmap                
      0.0            8,311          3       2,770.3      2,743.0     1,750        3,818       1,034.3  fcntl                 
      0.0            7,399          1       7,399.0      7,399.0     7,399        7,399           0.0  connect               
      0.0            3,260          2       1,630.0      1,630.0     1,284        1,976         489.3  fflush                
      0.0            1,550          1       1,550.0      1,550.0     1,550        1,550           0.0  bind                  
      0.0            1,502          1       1,502.0      1,502.0     1,502        1,502           0.0  pthread_cond_broadcast

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  ------------  ----------------------
     84.3       97,391,992          3  32,463,997.3      48,537.0      46,205  97,297,250  56,147,243.8  cudaMalloc            
     13.1       15,086,180          1  15,086,180.0  15,086,180.0  15,086,180  15,086,180           0.0  cuLibraryLoadData     
      2.3        2,639,864          1   2,639,864.0   2,639,864.0   2,639,864   2,639,864           0.0  cudaDeviceSynchronize 
      0.3          359,360          3     119,786.7     122,280.0     111,579     125,501       7,288.2  cudaFree              
      0.1           79,716          1      79,716.0      79,716.0      79,716      79,716           0.0  cudaLaunchKernel      
      0.0              717          1         717.0         717.0         717         717           0.0  cuKernelGetName       
      0.0              526          1         526.0         526.0         526         526           0.0  cuModuleGetLoadingMode
      0.0              258          1         258.0         258.0         258         258           0.0  cuLibraryGetKernel    

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                            Name                          
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  --------------------------------------------------------
    100.0        2,638,124          1  2,638,124.0  2,638,124.0  2,638,124  2,638,124          0.0  matrixMultiply(float *, float *, float *, int, int, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain GPU memory data.
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain GPU memory data.
Generated:
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.nsys-rep
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite
```


3. 
```bash
nvcc -o 02 02_tiled_matmul.cu
nsys profile --stats=true ./02
```

```bash
> sudo nsys profile --stats=true ./02
Collecting data...
Generating '/tmp/nsys-report-1bef.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     72.6      287,028,555         11  26,093,505.0  8,165,760.0   233,639  213,036,338  62,839,640.3  poll                  
     26.7      105,445,161        460     229,228.6      8,518.5     1,093   12,621,085     807,734.9  ioctl                 
      0.3        1,350,546         25      54,021.8      8,700.0     5,723      880,946     173,815.8  mmap64                
      0.1          587,837          9      65,315.2     55,617.0    11,497      216,278      60,219.0  sem_timedwait         
      0.1          215,478         43       5,011.1      4,188.0     2,372       10,872       1,835.1  open64                
      0.0          129,169         28       4,613.2      3,297.0     1,014       15,950       3,653.7  fopen                 
      0.0          127,223          3      42,407.7     41,549.0    32,649       53,025      10,215.1  pthread_create        
      0.0          124,071         13       9,543.9      5,102.0     1,930       48,062      12,379.8  mmap                  
      0.0           63,247          1      63,247.0     63,247.0    63,247       63,247           0.0  pthread_cond_wait     
      0.0           36,071          8       4,508.9      3,022.0     1,241       12,502       3,995.5  close                 
      0.0           31,965         18       1,775.8      1,443.5     1,001        4,618         905.5  fclose                
      0.0           27,534          1      27,534.0     27,534.0    27,534       27,534           0.0  fgets                 
      0.0           21,848          5       4,369.6      2,595.0     1,052       10,848       4,006.6  fread                 
      0.0           20,642          6       3,440.3      3,327.0     1,085        6,293       1,820.0  open                  
      0.0           12,734          3       4,244.7      3,627.0     3,157        5,950       1,495.4  munmap                
      0.0           11,550          7       1,650.0      1,311.0     1,041        2,688         688.5  write                 
      0.0           11,003          8       1,375.4      1,031.5     1,001        2,354         550.9  read                  
      0.0            9,961          3       3,320.3      4,044.0     1,459        4,458       1,625.2  pipe2                 
      0.0            9,257          2       4,628.5      4,628.5     3,618        5,639       1,429.1  socket                
      0.0            6,432          1       6,432.0      6,432.0     6,432        6,432           0.0  connect               
      0.0            2,562          1       2,562.0      2,562.0     2,562        2,562           0.0  fcntl                 
      0.0            2,094          1       2,094.0      2,094.0     2,094        2,094           0.0  fwrite                
      0.0            1,954          1       1,954.0      1,954.0     1,954        1,954           0.0  bind                  
      0.0            1,497          1       1,497.0      1,497.0     1,497        1,497           0.0  pthread_cond_broadcast

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  ---------  ----------  ------------  ----------------------
     97.4       94,815,271          3  31,605,090.3     47,144.0     46,725  94,721,402  54,660,329.3  cudaMalloc            
      2.1        2,001,177          1   2,001,177.0  2,001,177.0  2,001,177   2,001,177           0.0  cudaDeviceSynchronize 
      0.3          307,669          3     102,556.3    118,765.0     65,066     123,838      32,566.5  cudaFree              
      0.1          140,497          1     140,497.0    140,497.0    140,497     140,497           0.0  cuLibraryLoadData     
      0.1           59,709          1      59,709.0     59,709.0     59,709      59,709           0.0  cudaLaunchKernel      
      0.0              589          1         589.0        589.0        589         589           0.0  cuModuleGetLoadingMode
      0.0              417          1         417.0        417.0        417         417           0.0  cuLibraryGetKernel    
      0.0              227          1         227.0        227.0        227         227           0.0  cuKernelGetName       

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                Name                               
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -----------------------------------------------------------------
    100.0        2,001,006          1  2,001,006.0  2,001,006.0  2,001,006  2,001,006          0.0  matrixMultiplyOptimized(float *, float *, float *, int, int, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain GPU memory data.
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
SKIPPED: /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite does not contain GPU memory data.
Generated:
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.nsys-rep
        /home/andy/Workplace/projects/cuda-course/05_Writing_your_First_Kernels/03_Profiling/report1.sqlite
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