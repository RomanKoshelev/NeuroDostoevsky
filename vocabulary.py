import numpy as np
import nltk
from nltk import FreqDist
import pickle

UNK, UNK_CODE = '<UNK>', 0
BOS, BOS_CODE = '<BOS>', 1
EOS, EOS_CODE = '<EOS>', 2
PAD, PAD_CODE = '<PAD>', 3

class Vocabulary:
    def __init__(self):
        self._tokens_to_words = None
        self._words_to_tokens = None
        
    def __str__(self):
        return "%s:\n  size: %d\n  _tokens_to_words: %s" % (self.__class__.__name__,self.size, self._tokens_to_words[:20])   

    def _update(self, fdist, path):
        with open(path, 'r') as f:
            text = f.read()
            text = text.replace('\xa0', ' ').replace('\ufeff','')
            text = text.lower()

        for sentence in nltk.tokenize.sent_tokenize(text):
            for word in nltk.tokenize.word_tokenize(sentence):
                fdist[word]+=1
        return fdist

    @property
    def size(self):
        return len(self._tokens_to_words)
    
    def build(self, paths, max_size=30000):
        if type(paths) is str:
            paths = [paths]

        fdist = FreqDist()
        for p in paths:
            fdist = self._update(fdist, p)
            
        most_common = fdist.most_common(max_size)
        words       = [ UNK, BOS, EOS, PAD ] + [w for w, _ in most_common]
        self._tokens_to_words = words
        self._words_to_tokens = {words[i]:i for i in range(len(words))}        

        
    def save(self, path):
        pickle.dump([self._tokens_to_words, self._words_to_tokens], open(path, "wb"))

        
    def restore(self, path):
        [self._tokens_to_words, self._words_to_tokens] = pickle.load(open(path, "rb"))
        
        
    def to_tokens(self, words):
        return [self._words_to_tokens.get(w, UNK_CODE) for w in words]

    
    def to_words(self, tokens):
        return [self._tokens_to_words[t] for t in tokens]
