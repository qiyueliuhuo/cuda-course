# 安装说明

1. 先运行：
```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove
```
然后前往 NVIDIA 官方下载页面：[downloads](https://developer.nvidia.com/cuda-downloads)

2. 在页面中选择与用于本课程的设备匹配的选项：
   - 操作系统（Operating System）
   - 架构（Architecture）
   - 发行版（Distribution）
   - 版本（Version）
   - 安装包类型（Installer Type）

3. 在“runfile”部分，你需要运行一条与下面非常相近的命令：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

4. 完成后，你应该可以运行 `nvcc --version` 来查看 NVIDIA CUDA 编译器的信息（版本等）。
   还可以运行 `nvidia-smi` 来确认 NVIDIA 识别了你的 CUDA 版本和已连接的 GPU。

5. 如果 `nvcc` 无法使用，先运行 `echo $SHELL`。  
   如果返回是 `/bin/bash`，把下面几行添加到 `~/.bashrc` 文件；如果是 `/bin/zsh`，则添加到 `~/.zshrc` 文件：

```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

   添加后运行 `source ~/.zshrc` 或 `source ~/.bashrc`，然后再次尝试 `nvcc -V`。

## 另外一种方式

- 也可运行本目录下的脚本：`./cuda-installer.sh`

## 关于 WSL2 的参考

有关在 WSL2 上干净安装 Ubuntu、CUDA、cuDNN 和 PyTorch 的文档，请参考这篇文章：
https://medium.com/@omkarpast/technical-documentation-for-clean-installation-of-ubuntu-cuda-cudnn-and-pytorch-on-wsl2-9b265a4b8821