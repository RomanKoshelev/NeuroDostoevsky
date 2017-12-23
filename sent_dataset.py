import numpy as np
import nltk

UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'
PAD = '_'

IGNORE = "()[]{}:<>~@#$%^/\|_+*…–«»"
AS_DOT = ";"

class SentenceDataset:
    def __init__(self, min_len, max_len):
        self.max_len       = max_len
        self.min_len       = min_len
        self.sentences     = []
        self.word_to_token = {}
        self.token_to_word = {}
        self.num_tokens    = 0
    
    
    def load(self, path, size = None):
        with open(path, 'r') as f:
            text = f.read()
            if size is not None:
                text = text[:size]
            text = text.replace('\xa0', ' ').replace('\ufeff','')

        for c in AS_DOT:
            text=text.replace(c, '. ')
            
        sents = []
        def prepare(s):
            s = s.lower()
            for c in IGNORE:
                s=s.replace(c, ' ')
            return s
        
        for line in text.split('\n'):
            ls = nltk.tokenize.sent_tokenize(line)
            sents.extend([prepare(s) for s in ls])
        
        vocab   = set([UNK, BOS, EOS, PAD])
        w_sents = []
        for sent in sents:
            words = [w for w in nltk.tokenize.word_tokenize(sent)]
            vocab.update(words)
            if len(words) < self.min_len or len(words) > self.max_len:
                continue
            w_sents.append(words)

        self.num_tokens    = len(vocab)
        self.word_to_token = {w: i for i, w in enumerate(vocab)}
        self.token_to_word = dict(enumerate(vocab))
        
        self.sentences = np.zeros([len(w_sents), self.max_len+2], dtype=np.int32)
        for i in range(len(w_sents)):
            self.sentences[i] = self.encode(w_sents[i])


    def get_data_size(self):
        return len(self.sentences)
    
        
    def encode(self, words):
        words = words[:self.max_len]
        npads = self.max_len - len(words)
        words = [BOS] + words + [EOS] + [PAD] * npads
        return np.array([self.word_to_token[w] for w in words], dtype=np.int32)

    
    def decode(self, tokens):
        text = " ".join([self.token_to_word[t] for t in tokens])
        return text

    
    def get_next_batch(self, bs):
        num   = len(self.sentences)
        idx   = np.random.choice(np.arange(num), bs, replace=bs>num)
        sents = self.sentences[idx]
        return sents