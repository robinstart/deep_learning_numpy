import numpy as np

class GDoptimizer():
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        
    def minimize(self, layers):
        for layer in layers:
            for variable, gradient in zip(layer.variable, layer.gradient):
                variable -= self.learning_rate * gradient

class Momentum_optimizer():
    """
    learning_rate: (default: 0.001)
    momentum: (default: 0.9)
    """
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = []
        self.iteration = 0
        
    def minimize(self, layers):
        if len(self.v) == 0:
            for layer in layers:
                self.v.append([0] * len(layer.variable))
                
        self.iteration += 1
        
        for layer, vs in zip(layers, self.v):
            for variable, gradient, v in zip(layer.variable, layer.gradient, vs):
                v = ((1 - self.momentum) * gradient + self.momentum * v) / (1 - self.momentum ** self.iteration)
                variable -= self.learning_rate * v
                
class Adagrad_optimizer():
    """
    learning_rate: (default: 0.001)
    """
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.s = []
        
    def minimize(self, layers):
        if len(self.s) == 0:
            for layer in layers:
                self.s.append([0] * len(layer.variable))
                
        for layer, ss in zip(layers, self.s):
            for variable, gradient, s in zip(layer.variable, layer.gradient, ss):
                s += gradient ** 2
                variable -= self.learning_rate * gradient / (s + self.epsilon) ** 0.5

class RMSprop_optimizer():
    """
    learning_rate: (default: 0.001)
    decay: (default: 0.999)
    """
    def __init__(self, learning_rate=0.001, decay=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.s = []
        self.iteration = 0
        
    def minimize(self, layers):
        if len(self.s) == 0:
            for layer in layers:
                self.s.append([0] * len(layer.variable))
                
        self.iteration += 1
        
        for layer, ss in zip(layers, self.s):
            for variable, gradient, s in zip(layer.variable, layer.gradient, ss):
                s = (self.decay * s + (1 - self.decay) * gradient ** 2)
                variable -= self.learning_rate * gradient / (s / (1 - self.decay ** self.iteration) + self.epsilon) ** 0.5
                
class Adam_optimizer():
    """
    learning_rate: (default: 0.001)
    momentum: (default: 0.9)
    decay: (default: 0.999)
    """
    def __init__(self, learning_rate=0.001, momentum=0.9, decay=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.epsilon = epsilon
        self.v = []
        self.s = []
        self.iteration = 0
        
    def minimize(self, layers):
        if len(self.v) == 0:
            for layer in layers:
                self.v.append([0] * len(layer.variable))
                self.s.append([0] * len(layer.variable))
        
        self.iteration += 1
                
        for layer, vs, ss in zip(layers, self.v, self.s):
            for variable, gradient, v, s in zip(layer.variable, layer.gradient, vs, ss):
                v = ((1 - self.momentum) * gradient + self.momentum * v) 
                s = (self.decay * s + (1 - self.decay) * gradient ** 2)
                variable -= self.learning_rate * (v / (1 - self.momentum ** self.iteration)) / (s / (1 - self.decay ** self.iteration)  + self.epsilon) ** 0.5
                
                
                