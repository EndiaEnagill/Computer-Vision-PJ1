import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        """
        epsilon: 一个小的正数, 用于避免出现log(0), log(1), 除零的情况
        y_pred: 样本预测值, 存储各个样本对各个类别的预测分数
        y_true: 样本的真实标签
        """
        self.epsilon = 1e-12
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred: 样本预测值, 存储各个样本对各个类别的预测分数
        y_true: 样本的真实标签
        """
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true
        loss = - np.mean(np.sum(np.eye(10)[self.y_true] * np.log(self.y_pred), axis = 1))
        return loss
    
    def backward(self):
        #grad = - np.eye(10)[self.y_true] / ((self.y_pred) * self.y_true.shape[0])
        grad = (self.y_pred - np.eye(10)[self.y_true]) / self.y_pred.shape[0]
        return grad