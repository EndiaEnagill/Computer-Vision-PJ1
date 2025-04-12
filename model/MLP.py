import numpy as np
import model.Layer as Layer

import pickle

class MLP:
    def __init__(self, arc = None):
        """
        arc: MLP的隐藏层大小设置和激活函数, 为字典类型列表, 元素设置如下
                layer: {
                    'input_dim': int
                    'output_dim': int
                    'activation': str
                }
        """
        self.layers = []
        self.arc = arc if arc else []
        for layer in self.arc:
            self.layers.append(Layer.Linear(layer['input_dim'], layer['output_dim']))
            self.layers.append(Layer.Activation(layer['activation']))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, grad):
        """
        grad: 上游梯度
        layer.input_cache: 记录该层的输入值
        """
        for layer in reversed(self.layers):
            if isinstance(layer, Layer.Linear):
                x = layer.input_cache
                dW, db, dx = layer.backward(x, grad)
                layer.dW = dW
                layer.db = db
                grad = dx
            if isinstance(layer, Layer.Activation):
                z = layer.input_cache
                grad = layer.backward(z, grad)
        return grad
    
    def predict(self, A):
        return np.argmax(self.forward(A), axis = 1)

    def copy(self):
        model_copy = MLP(self.arc)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer.Linear):
                model_copy.layers[i].W = layer.W.copy()
                model_copy.layers[i].b = layer.b.copy()
            if isinstance(layer, Layer.Activation):
                model_copy.layers[i].activation_type = layer.activation_type
        return model_copy
    
    def save_model(self, path):
        """
        保存模型的构架以及各层的参数W, b, 激活函数类型。
        """
        model_para = {}
        model_para['arc'] = self.arc
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer.Linear):
                model_para[f'Linear_{i}_W'] = layer.W
                model_para[f'Linear_{i}_b'] = layer.b
            if isinstance(layer, Layer.Activation):
                model_para[f'Act_fun_{i}'] = layer.activation_type
        with open(path, 'wb') as f:
            pickle.dump(model_para, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            model_para = pickle.load(f)
        self.arc = model_para['arc']
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer.Linear):
                layer.W = model_para[f'Linear_{i}_W']
                layer.b = model_para[f'Linear_{i}_b']
                layer.zero_grad()
            if isinstance(layer, Layer.Activation):
                layer.activation_type = model_para[f'Act_fun_{i}']