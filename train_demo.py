from model.Dataloader import cifar10
from model.Loss import CrossEntropyLoss
from model.MLP import MLP
from model.SGDoptimizer import SGD
from model.Train import Trainer

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

def main():
    dataloader = cifar10(**dataloader_para)
    model = MLP(arc)
    optimizer = SGD(**SGD_para)
    loss = CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss, dataloader, **trainer_para)
    print('training......')
    trainer.train(save_model_cache = 'demo', versible = False)
    print('train is over.')
    trainer.save_log('./model_data/demo')
    print('save log over')
    trainer.save_best_model('./model_data/demo')
    print('save model over')
    trainer.train_path(dir = './model_data/demo', epoch = trainer_para['num_epochs'])
    print('save train path over')
    trainer.valid_path(dir = './model_data/demo', epoch = trainer_para['num_epochs'])
    print('save valid path over')
    trainer.loss_plot(dir = './model_data/demo', epoch = trainer_para['num_epochs'])
    print('save loss over')
    trainer.acc_plot(dir = './model_data/demo', epoch = trainer_para['num_epochs'])
    print('save acc over')

    trainer.clear_cache()

if __name__ == "__main__":
    main()