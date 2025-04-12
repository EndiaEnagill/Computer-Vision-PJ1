import numpy as np

class Layer:
    def __init__(self):
        self.input_cache = None

class Activation(Layer):
    """
    activation_type: 激活函数类型，包含以下几种可选函数。
        sigmoid
        relu
        leakyrelu(aplha = 0.01)
        softmax
    """
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def forward(self, z):
        self.input_cache = z
        ac_type = self.activation_type
        if ac_type == 'sigmoid':
            A = 1 / (1 + np.exp(-z))
        elif ac_type == 'relu':
            A = np.maximum(0, z)
        elif ac_type == 'leakyrelu':
            A = np.where(z > 0, z, z * 0.01)
        elif ac_type == 'softmax':
            A = np.exp(z - np.max(z, axis = 1, keepdims = True))
            A = A / np.sum(A, axis = 1, keepdims = True)
        return A
    
    def backward(self, z, dA):
        """
        dA: 上游梯度
        """
        ac_type = self.activation_type
        if ac_type == 'sigmoid':
            z = self.forward(z)
            dZ = dA * z * (1 - z)
        elif ac_type == 'relu':
            dZ = dA * (z > 0)
        elif ac_type == 'leakyrelu':
            dZ = dA * np.where(z >= 0, 1, 0.01)
        elif ac_type == 'softmax':
            #z = self.forward(z)
            #dZ = z - dA 
            s = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
            dZ = s * (dA - np.sum(dA * s, axis=0, keepdims=True))
        return dZ
    
class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        # 从均值为0, 标准差为1/sqrt(input_dim)的正态分布采样权重
        # 防止输入维度较大时，权重值过大导致梯度爆炸或激活值饱和
        self.W = np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, x, dZ):
        """
        dZ: 上游梯度
        """
        dW = np.dot(x.T, dZ)
        db = np.sum(dZ, axis = 0, keepdims = True)
        dx = np.dot(dZ, self.W.T)
        return dW, db, dx
    
    def zero_grad(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)