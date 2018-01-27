import numpy as np
import nltk
from utils import sent_to_words, make_dir
from vocabulary import BOS_CODE, EOS_CODE, PAD_CODE
import pickle

class TextDataset:
    def __init__(self):
        self._data       = None
        self._data_limit = None
        self._path       = None
    
    def __str__(self):
        return "%s:\n  path: %s\n  shape: %s\n  data_limit: %s" % (
            self.__class__.__name__, self._path, self.shape, self._data_limit)
    
    def _read_text(self, path, voc, min_len, max_len):
        with open(path, 'r') as f:
            text = f.read()
            text = text.replace('\xa0', ' ').replace('\ufeff','')
            text = text.lower()
        tokenized_text = []
        
        for sentence in nltk.tokenize.sent_tokenize(text):
            words = sent_to_words(sentence)
            if min_len <= len(words) <= max_len:
                tokens = voc.to_tokens(words)
                tokenized_text.append(tokens)
        return tokenized_text

    @property
    def shape(self):
        return list(self._data.shape)

    def get_data_size(self):
        return self._data_limit or self._data.shape[0]
    
    def get_seq_len(self):
        return self._data.shape[1]
    
    
    def build(self, paths, vocab, max_len, min_len=1):
        if type(paths) is str:
            paths = [paths]
        
        sentences = []
        
        for p in paths:
            sentences.extend(self._read_text(p, vocab, min_len, max_len))

        def to_data(sent):
            assert min_len <= len(sent) <= max_len
            npads = max_len - len(sent)
            sent = [BOS_CODE] + sent + [EOS_CODE] + [PAD_CODE] * npads
            assert len(sent) == max_len + 2
            return np.array(sent, dtype=np.int32)
            
        self._data = np.zeros([len(sentences), max_len+2], dtype=np.int32)
        for i in range(len(sentences)):
            self._data[i] = to_data(sentences[i])
            
            
    def save(self, path):
        make_dir(path)
        pickle.dump([self._data], open(path, "wb"))
        self._path = path 

    def restore(self, path):
        [self._data] = pickle.load(open(path, "rb"))
        self._path = path 

    
    def set_data_limit(self, limit):
        self._data_limit = limit

    
    def get_next_batch(self, bs):
        num   = self._data_limit or len(self._data)
        idx   = np.random.choice(np.arange(num), bs, replace=bs>num)
        sents = self._data[idx]
        return sents
