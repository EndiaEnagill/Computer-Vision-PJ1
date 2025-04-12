import numpy as np

def cartesian(*para):
    """
    生成给定参数的笛卡尔积, 用于参数查找, 顺序如下:
    第一层的output, 第二层的output, activation_type
    batch_size, n_valid
    learning_rate, threshold, reg, decay_rate, decay_freq
    num_epochs, eval_step
    """
    combinations = [[]]
    for item in range(len(para)):
        sub_c = []
        for s in combinations:
            for i in para[item]:
                sub_c.append(s + [i])
        combinations = sub_c
    return combinations

def generate_para(c):
    arc = [
        {'input_dim': 3072, 'output_dim': c[0], 'activation': c[2]},
        {'input_dim': c[0], 'output_dim': c[1], 'activation': c[2]},
        {'input_dim': c[1], 'output_dim': 10, 'activation': c[2]}
    ]
    dataloader_para = {
        'root_dir': './cifar10',
        'batch_size': c[3],
        'normalize': True,
        'n_valid': c[4]
    }
    SGD_para = {
        'learning_rate': c[5],
        'threshold': c[6],
        'reg': c[7],
        'decay_rate': c[8],
        'decay_freq': c[9]
    }
    trainer_para = {
        'num_epochs': c[10],
        'eval_step': c[11]
    }
    return {'arc': arc, 'dataloader_para': dataloader_para, 'SGD_para': SGD_para, 'trainer_para': trainer_para}
