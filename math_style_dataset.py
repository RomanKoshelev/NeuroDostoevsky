import numpy as np
import nltk
from utils import sent_to_words, make_dir
from math_dataset import MathDataset
from vocabulary import BOS_CODE, EOS_CODE, PAD_CODE
import pickle

class MathStyleDataset:
    def __init__(self):
        self._ds0 = MathDataset()
        self._ds1 = MathDataset()
        
    def __str__(self):
        return "%s:\n  path_0: %s\n  path_1: %s\n  shape: %s\n" % (
            self.__class__.__name__, self._ds0._path, self._ds1._path, self.shape)
        
    def build(self, path, vocab, max_len, min_len=1):
        self._ds0.build(path % 0, vocab, max_len, min_len)
        self._ds1.build(path % 1, vocab, max_len, min_len)
        assert self._ds0.shape[1] == self._ds1.shape[1]

    @property
    def shape(self):
        return [self._ds0.shape[0]+self._ds0.shape[0], self._ds0.shape[1]]

    def get_data_size(self):
        return self.shape[0]
    
    def get_seq_len(self):
        return self.shape[1]
            
    def get_next_batch(self, bs):
        bs0 = bs//2
        bs1 = bs - bs0
        
        sent0 = self._ds0.get_next_batch(bs0)
        sent1 = self._ds1.get_next_batch(bs1)
        sents = np.concatenate([sent0,sent1], axis=0)
        
        styles = np.concatenate([np.zeros(bs0), np.ones(bs1)], axis=0)
        
        return sents, styles
    
    def save(self, path):
        self._ds0.save(path % 0)
        self._ds1.save(path % 1)

    def restore(self, path):
        self._ds0.restore(path % 0)
        self._ds1.restore(path % 1)