# 自定义 PyTorch 扩展 (Custom PyTorch Extensions)

```bash
python setup.py install 
```

## 什么是 `scalar_t` 类型?
- 可以将其理解为 CUDA torch 张量 (tensor) 中元素的类型。
- 它会被安全地编译为适合 GPU 的对应类型（如 fp32 或 fp64）。

## 为什么要使用 `__restrict__`?

```cpp
// 考虑以下代码的行为方式

void add_arrays(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] + b[i];
    }
}

int main() {
    int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 重叠调用 (Overlapping call)
    // a 指向 data[0]，b 指向 data[3]
    add_arrays(data, data + 3, 7);
    
    // 打印结果
    for (int i = 0; i < 10; i++) {
        printf("%d ", data[i]);
    }
    return 0;
}
```

```python
# 'data' 数组的初始状态:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 内存布局可视化:
#  a (data)     b (data + 3)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#  ^        ^
#  |        |
#  a[0]     b[0]

# i = 0 之后: data[0] = data[0] + data[3]
[5, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ^

# i = 1 之后: data[1] = data[1] + data[4]
[5, 7, 3, 4, 5, 6, 7, 8, 9, 10]
#    ^

# i = 2 之后: data[2] = data[2] + data[5]
[5, 7, 9, 4, 5, 6, 7, 8, 9, 10]
#       ^

# i = 3 之后: data[3] = data[3] + data[6]
[5, 7, 9, 11, 5, 6, 7, 8, 9, 10]
#          ^

# i = 4 之后: data[4] = data[4] + data[7]
# 注意: data[4] 此时的值已经发生了改变！
[5, 7, 9, 11, 13, 6, 7, 8, 9, 10]
#              ^

# i = 5 之后: data[5] = data[5] + data[8]
[5, 7, 9, 11, 13, 15, 7, 8, 9, 10]
#                  ^

# i = 6 之后: data[6] = data[6] + data[9]
[5, 7, 9, 11, 13, 15, 17, 8, 9, 10]
#                      ^

# 'data' 数组的最终状态:
data = [5, 7, 9, 11, 13, 15, 17, 8, 9, 10]
```

## Torch 绑定部分 (Torch Binding section)
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polynomial_activation", &polynomial_activation_cuda, "Polynomial activation (CUDA)");
}
```

本部分使用 pybind11 为 CUDA 扩展创建一个 Python 模块：
- `PYBIND11_MODULE` 是一个宏 (macro)，用于定义 Python 模块的入口点 (entry point)。
- `TORCH_EXTENSION_NAME` 是由 PyTorch 定义的宏，它会被展开为扩展的名称（通常源自 `setup.py` 文件）。
- `m` 是正在创建的模块对象。
- `m.def()` 用于向模块添加一个新函数：
  - 第一个参数 `"polynomial_activation"` 是该函数在 Python 中的名称。
  - `&polynomial_activation_cuda` 是指向要调用的 C++ 函数的指针。
  - 最后一个参数是函数的文档字符串 (docstring)。

> 通过使用 `__restrict__`，我们实质上是在告诉编译器这些数组**不存在重叠 (not overlapping)**。
> 这样一来，编译器就可以对内存布局做出假设，从而进行**激进的优化 (aggressively optimize)**。

- 请注意顶行显示该内容被保存到了 `/home/elliot/.cache/torch_extensions/py311_cu121`（如果 `.cache` 目录被二进制文件塞满，你可以清理它）。


## 学习资源 (Learning Resources)
- https://github.com/pytorch/extension-cpp
- https://pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pytorch.org/docs/stable/notes/extending.html
