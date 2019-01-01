import numpy as np
import os
import gzip as gz
import pickle as pk
import pandas as pd

class mnist():
    def __init__(self):
        self.test_images, self.test_labels, self.train_images, self.train_labels = self.load_mnist()
        self.train_images = self.train_images.reshape(60000, -1)
        self.test_images = self.test_images.reshape(10000, -1)
        self.train_labels = np.array(pd.get_dummies(self.train_labels))
        self.test_labels = np.array(pd.get_dummies(self.test_labels))
    
    def load_mnist(self):
        """
        load from gzip file
        output: test_images, test_labels, train_images, train_labels
        """
        file_names = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
                      "train-labels-idx1-ubyte.gz" ]
        paths = [os.path.join(os.getcwd(), "mnist", file_name) for file_name in file_names]
        data = []
        for i, path in enumerate(paths):
            with gz.open(path) as b:
                if i % 2:
                    data.append(np.frombuffer(b.read(), dtype=np.uint8, offset=8))
                else:
                    data.append(np.frombuffer(b.read(), dtype=np.uint8, offset=16))
        return data
    
    def _trans_to_image(self, data):
        """
        (batch, 28 * 28) -> (batch, 28, 28)
        """
        return data.reshape(-1, 28, 28)
    
    @property
    def train_2d_images(self):
        """
        train images transformed to 2d
        """
        return self._trans_to_image(self.train_images)
    
    @property
    def test_2d_images(self):
        """
        test images transformed to 2d
        """
        return self._trans_to_image(self.test_images)
    