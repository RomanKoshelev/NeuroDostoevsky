import numpy as np
from math_dataset import MathDataset

class MathDiscrDataset:
    def __init__(self, pos_path, neg_path):
        pos = MathDataset()
        neg = MathDataset()
        pos.restore(pos_path)
        neg.restore(neg_path)
        
        assert pos.shape[1] == neg.shape[1]
        
        self._data   = np.zeros([pos.shape[0] + neg.shape[0], pos.shape[1]], dtype=np.int32)
        self._labels = np.zeros([self.shape[0]], dtype=np.int32)
        self._data[:pos.shape[0]] = pos._data
        self._data[pos.shape[0]:] = neg._data
        self._labels[:pos.shape[0]] = 1
    
    def __str__(self):
        return "%s:\n  shape: %s" % (self.__class__.__name__, self.shape)
    
    @property
    def shape(self):
        return list(self._data.shape)    
    
    def get_data_size(self):
        return self.shape[0]
    
    def get_seq_len(self):
        return self.shape[1]
            
    def get_next_batch(self, bs):
        num    = self.shape[0]
        idx    = np.random.choice(np.arange(num), bs, replace=bs>num)
        sents  = self._data[idx]
        labels = self._labels[idx]
        return sents, labels
