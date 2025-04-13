### 手工搭建三层神经网络分类器

---

#### 任务描述

手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

1. 本次作业要求自主实现反向传播，**不允许使用 pytorch，tensorflow** 等现成的支持自动微分的深度学习框架，**可以使用 numpy**；
2. 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计。
3. 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。

#### 设置

数据集：CIFAR10

环境：python 3.10.10及更高版本。需要以下包进行模型的训练，保存以及可视化：

```
pip install numpy pickle matplotlib csv
```

#### 模型训练与测试

运行 `train_demo.py` 以训练一个预设好超参数的模型。

运行`test.py` 可以加载指定的model_data中的某个模型参数，并在验证集上进行测试。

如果想要测试模型在不同超参数下的性能，请编辑 `Grid_Search.py` 中的超参数列表。运行代码，将会统计所有超参数组合，并按顺序依次训练模型。训练好的模型将保存在model文件夹下，每个模型有对应的唯一文件夹用于存储模型的训练日志 `train_log.txt`，训练得到的最佳模型参数`best_model.pkl`，以及训练过程中损失及准确率的可视化展示。

在训练过程中，你可以在终端查看训练进度：

```
training iter 10......
training for epoch 1......
training for epoch 2......
```

在每个模型训练完成后，将会显示模型保存进度：

```
train is over.
save log over.
save model over.
save train path over.
save valid path over.
save loss over.
save acc over.
```

使用 `Grid_Search.py` 训练模型时，训练结果将会保存在表格 `output.csv` 中。

历史的训练结果全部保存在model_data中，你可以按需下载。

