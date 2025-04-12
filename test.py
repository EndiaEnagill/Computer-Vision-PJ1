from model.Dataloader import cifar10
from model.Loss import CrossEntropyLoss
from model.MLP import MLP
from model.SGDoptimizer import SGD
import numpy as np
import model.Layer as Layer
from model.Test import tester


arc = [
    {'input_dim': 3072, 'output_dim': 128, 'activation': 'relu'},
    {'input_dim': 128, 'output_dim': 64, 'activation': 'relu'},
    {'input_dim': 64, 'output_dim': 10, 'activation': 'relu'}
]

# 数据加载参数
dataloader_para = {
    'root_dir': './cifar10',
    'batch_size': 16,
    'normalize': True,
    'n_valid': 5000
}

# 优化器参数
SGD_para = {
    'learning_rate': 0.01,
    'threshold': 1e-5,
    'reg': 0.001,
    'decay_rate': 0.95,
    'decay_freq': 5000
}

# 训练器参数
trainer_para = {
    'num_epochs': 100,
    'eval_step': 5
}

dataloader = cifar10(**dataloader_para)
model = MLP(arc)
model.load_model('./model_data/demo/best_model.pkl')
optimizer = SGD(**SGD_para)
Loss = CrossEntropyLoss()

test_loss, test_acc = tester(model, Loss, dataloader).evaluate()
print(test_loss)
print(test_acc)


