from model.Dataloader import cifar10
from model.Loss import CrossEntropyLoss
from model.MLP import MLP
from model.SGDoptimizer import SGD
from model.Train import Trainer
from model.Test import tester
from model.GridSearch import cartesian, generate_para

import csv
import os

# MLP相关参数
first_input_dim = [128]
second_input_dim = [64]
activation_type = ['softmax']

# 数据加载器参数
batch_size = [16] # [4, 16, 32]
n_valid = [5000]

# 优化器参数
learning_rate = [0.01] # [0.01, 0.005]
threshold = [1e-5]
reg = [0.01] # [0.01, 0.001]
decay_rate = [0.95] # [0.9, 0.95]
decay_freq = [5000]

# 训练器参数
num_epochs = [120]
eval_step = [5]

combinations = cartesian(first_input_dim, second_input_dim, activation_type, batch_size, n_valid, learning_rate, threshold, reg, decay_rate, decay_freq, num_epochs, eval_step)
print(combinations)
file_exists = os.path.exists('output.csv')
with open('output.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['first_input_dim', 'second_input_dim', 'activation_type', 'batch_size', 'n_valid', 'learning_rate', 'threshold', 'reg', 'decay_rate', 'decay_freq', 'num_epochs', 'eval_step', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc', 'ave_time'])  # 写入表头
    for i, combination in enumerate(combinations):
        iter = i + 10
        para = generate_para(combination)
        arc = para['arc']
        dataloader_para = para['dataloader_para']
        SGD_para = para['SGD_para']
        trainer_para = para['trainer_para']

        dataloader = cifar10(**dataloader_para)
        model = MLP(arc)
        optimizer = SGD(**SGD_para)
        loss = CrossEntropyLoss()

        # 训练并保存数据
        trainer = Trainer(model, optimizer, loss, dataloader, **trainer_para)
        print(f'training iter {iter}......')
        trainer.train(save_model_cache = f'{iter}', versible = False)
        print('train is over.')
        trainer.save_log(f'./model_data/{iter}')
        print('save log over')
        trainer.save_best_model(f'./model_data/{iter}')
        print('save model over')
        trainer.train_path(dir = f'./model_data/{iter}', epoch = trainer_para['num_epochs'])
        print('save train path over')
        trainer.valid_path(dir = f'./model_data/{iter}', epoch = trainer_para['num_epochs'])
        print('save valid path over')
        trainer.loss_plot(dir = f'./model_data/{iter}', epoch = trainer_para['num_epochs'])
        print('save loss over')
        trainer.acc_plot(dir = f'./model_data/{iter}', epoch = trainer_para['num_epochs'])
        print('save acc over')

        # 保存模型参数作为标识
        with open(f'./model_data/{iter}/train_para.txt', 'w') as f:
            for i in para:
                f.write(str(para))
        print('save para over')

        test_loss, test_acc = tester(model, loss, dataloader).evaluate()

        # 向csv中写入训练结果
        row = combination + [trainer.train_loss[-1], trainer.train_acc[-1], trainer.valid_loss[-1], trainer.valid_acc[-1], test_loss, test_acc, sum(trainer.time)/len(trainer.time)]
        writer.writerow(row)

        # 清除缓存
        trainer.clear_cache()