import numpy as np

class sigmoid():
    def __init__(self):
        self.out = None
        
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out # if return self.out it can be change or remove
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out) # dout * y * (1 - y)

class relu():
    def __init__(self):
        self.mask = None
        
    def forward(self, X):
        self.mask = (X >= 0).astype(np.float32)
        return X * self.mask
    
    def backward(self, dout):
        return (dout * self.mask)
