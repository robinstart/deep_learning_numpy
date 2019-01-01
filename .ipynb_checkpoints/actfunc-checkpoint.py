import numpy as np

class sigmoid():
    def __init__(self):
        self.out = None
        self.initializer = Xavier_initializer
        
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out # if return self.out it can be change or remove
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out) # dout * y * (1 - y)

class relu():
    def __init__(self):
        self.mask = None
        self.initializer = He_initializer
        
    def forward(self, X):
        self.mask = (X >= 0).astype(np.float32)
        return X * self.mask
    
    def backward(self, dout):
        return (dout * self.mask)

class learky_relu():
    def __init__(self, leak=0.1):
        self.leak = leak
        self.mask = None
        self.initializer = He_initializer
        
    def forward(self, X):
        self.mask = (X >= 0).astype(np.float32) + (X < 0).astype(np.float32) * self.leak
        return X * self.mask
    
    def backward(self, dout):
        return (dout * self.mask)
    

def He_initializer(input_size, output_size):
    return (2 / (input_size + output_size) ** 0.5)

def Xavier_initializer(input_size, output_size):
    return (2 / (input_size + output_size)) ** 0.5