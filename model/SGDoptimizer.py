import model.Layer as Layer

class SGD:
    def __init__(self, learning_rate = 0.01, threshold = 0.01,  reg= 0, decay_rate = 0, decay_freq = 1000):
        self.learning_rate = learning_rate
        self.threshold = threshold
        # L2正则化
        self.reg = reg 
        # 学习率下降
        self.decay_rate = decay_rate
        self.decay_freq = decay_freq
        self.iter = 0

    def step(self, model):
        """
        进行一次更新
        实现SGD优化和学习率下降
        """
        for layer in model.layers:
            if isinstance(layer, Layer.Linear):
                layer.W -= self.learning_rate * (layer.dW + 2 * self.reg * layer.W)
                layer.b -= self.learning_rate * (layer.db)
        
        self.iter += 1
        if self.decay_rate and self.iter % self.decay_freq == 0:
            if self.learning_rate > self.threshold:
                self.learning_rate *= self.decay_rate