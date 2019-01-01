import numpy as np

def get_batch(X, y, batchsize=64):
    """
    X: input
    y: target
    batchsize: (default: 64)
    output: tuple of batchsize record of input and batchsize record of target
    """
    mask = np.random.choice(np.arange(X.shape[0]), batchsize, replace=False)
    return X[mask], y[mask]

def numerial_gradient(func, X):
    """
    func: 
    X: 2d-array
    output: Numerial gradident
    Not recomend
    It's too slow when variables are large
    """
    epcilon = 1e-4
    grads = np.zeros_like(X)
    X_tmp = X
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = X[i, j] + epcilon
            y_plus = func(X)
            X[i, j] = X[i, j] - epcilon
            y_minus = func(X)
            grad = (y_plus - y_minus) / (2 * epcilon)
            grads[i, j] += grad
            X = X_tmp
            
    return grads

def one_hot_encoder(y):
    """
    when label is 1-d 
    y: 1-d array
    output: one_hot_encoded 2-d array, label_names 1-d array
    """
    if y.ndim != 1:
        print("input should be 1-d")
        return
    
    labels_name = np.unique(y)
    one_hot_encode = np.zeros((y.shape[0], len(labels_name)))
    for i, label in enumerate(labels_name):
        one_hot_encode[y == label, i] += 1
    
    return one_hot_encode, labels_name
