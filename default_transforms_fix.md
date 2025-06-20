# 修复 `default_transforms` 属性错误

## 问题原因

在处理训练数据集时，出现了以下错误：

```
AttributeError: 'YOLODataset' object has no attribute 'default_transforms'. Did you mean: 'build_transforms'?
```

这是因为在 `YOLODataset.get_labels()` 方法中使用了 `self.default_transforms` 属性，但是 `YOLODataset` 类没有定义这个属性。

出现这个问题的原因是：
1. 在 `get_labels()` 方法中，代码试图根据 `self.default_transforms` 的值来决定是否需要调整标签中的类别索引
2. 但是 `YOLODataset` 和它的父类 `BaseDataset` 都没有初始化这个属性

## 解决方案

我们采取了两种方法来解决这个问题：

### 1. 添加缺失的属性

在 `YOLODataset.__init__` 方法中添加了该属性的初始化：

```python
def __init__(self, *args, data=None, task="detect", **kwargs):
    # ... 现有代码 ...
    
    # 添加default_transforms属性
    self.default_transforms = False  # 默认不执行自动变换
    
    # ... 现有代码 ...
```

### 2. 删除不必要的代码

同时，我们删除了依赖这个属性的代码段，因为在我们的实现中不需要这种自动转换：

```python
# 删除前的代码
if self.default_transforms:
    for k in labels:
        if len(labels[k]):
            if (labels[k][:, 0] > 0).all():  # check if all classes > 0
                labels[k][:, 0] -= 1  # 调整类别索引
```

替换为：

```python
# 删除对default_transforms的检查，因为我们默认不做自动变换
# 如果需要，可以在子类中重写
```

## 这些更改的好处

1. **避免属性错误**: 通过添加缺失的属性，避免了 AttributeError
2. **简化代码**: 删除了不必要的代码块，简化了处理逻辑
3. **明确意图**: 通过注释明确表示我们默认不进行自动转换
4. **保持灵活性**: 允许子类根据需要重写此行为

这些修改使代码更加健壮，并确保数据处理逻辑的一致性。 