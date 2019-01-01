import numpy as np
from actfunc import Xavier_initializer

class multiply():
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y
    def backward(self, dout):
        dx = dout * y
        dy = dout * x
        return dx, dy
    
class add():
    def __init__(self):
        pass
        
    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        return dout, dout
    
class dense():
    """
    activation(X @ W + b)
    input_size: X is (batch_size, input_size) and W is (input_size, output_size)
    output_size: X is (batch_size, input_size) and W is (input_size, output_size)
    name: need for optimizer update variable
    activation: activation function (default: None)
    """
    def __init__(self, input_size, output_size, activation=None, name=None):
        self.b = np.zeros((1, output_size), dtype=np.float32)
        if activation is None:
            self.activation = activation
            self.W = np.random.randn(input_size, output_size).astype(np.float32) * Xavier_initializer(input_size, output_size)
        else:
            self.activation = activation()
            self.W = np.random.randn(input_size, output_size).astype(np.float32) * self.activation.initializer(input_size, output_size)
            
        self.name = name
        self.X = None
        self.dW = None
        self.db = None
    
    def forward(self, X): # activation(X @ W + b)
        """
        X: input
        output: activation(X @ W + b)
        """
        self.X = X
        a = self.X @ self.W + self.b
        if self.activation is None:
            return a
        else:
            return self.activation.forward(a)
    
    def backward(self, dout):
        """
        backpropagation
        dout: gradient of next layer
        output: dx
        """
        if self.activation is None:
            dactivation = dout
        else:
            dactivation = self.activation.backward(dout)
        self.dW = self.X.T @ dactivation
        self.db = np.sum(dactivation, axis=0, keepdims=True) # b was broadcasted to (batch_size, output_size), so need contraction
        return dactivation @ self.W.T # dX
        
    @property
    def variable(self):
        variables = []
        variables.append(self.W)
        variables.append(self.b)
        return variables
    
    @property
    def gradient(self):
        gradients = []
        gradients.append(self.dW)
        gradients.append(self.db)
        return gradients

class mean_square_error():
    """
    Mean Square Error
    (1 / 2) * (1 / n) * ((y - t) ** 2)
    """
    def __init__(self):
        self.y = None
        self.t = None
        self.size = None
        
    def forward(self, y, t):
        """
        y: y_predict
        t: target
        output: loss
        """
        if y.ndim == 1: # treat batch_size 1
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        self.y = y
        self.t = t
        self.size = np.product(y.shape)
        return np.sum((self.y - self.t) ** 2) * 0.5 / self.size # loss
    
    def backward(self, dout=1):
        return dout * (self.y - self.t) / self.size

class softmax_with_cross_entropy_error():
    """
    softmax + cross entropy error
    """
    def __init__(self):
        self.y = None
        self.t = None
        
    def forward(self, X, t):
        """
        X: (batch_size, labels) predict
        t: (batch_size, labels) target (need one hot encoding if target is 1-d array)
        """
        self.y = self.softmax(X)
        self.t = t
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss
        
    def backward(self, dout=1):
        """
        dout: (default: 1)
        output: dL/dX
        """
        return (self.y - self.t) / self.y.shape[0] # divided by batch_size
    
    def softmax(self, X):
        X_exp = np.exp(X - np.max(X, axis=1, keepdims=True)) # prevent overflow
        return X_exp / np.sum(X_exp, axis=1, keepdims=True)
        
    def cross_entropy_error(self, y, t):
        if y.ndim == 1: # treat batch_size 1
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        epcilon = 1e-7
        return -np.sum(t * np.log(y + epcilon)) / y.shape[0] # divided by batch_size
    
        