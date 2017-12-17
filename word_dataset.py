import numpy as np
import nltk
import re

class WordDataset:
    def __init__(self):
        self.text    = None
        self.encoded = []
    
    
    def load(self, path):
        STOP_WORDS   = [u"[", u"]"]
        PUNCTS       = [u".", u",", u"?", u"!"]
        MIN_SENT_LEN = 2
        MAX_SENT_LEN = 50
        
        with open(path, 'r') as f:
            text=f.read()
            text = text.replace(u'\xa0', u' ').replace(u'\ufeff','')

        sents = []
        for line in text.split('\n'):
            ls = nltk.tokenize.sent_tokenize(line)
            for s in ls:
                sents.extend(s.split(";"))

        word_text = []
        vocab     = set()
        for sent in sents:
            words = [w.lower() for w in nltk.tokenize.word_tokenize(sent) if w not in STOP_WORDS]
            if MIN_SENT_LEN <= len(words) <= MAX_SENT_LEN:
                if len(words) == 2 and words[1] in PUNCTS:
                    continue
                word_text.extend(words)
                vocab.update(words)

        self.char_text     = text
        self.word_text     = word_text
        self.word_to_token = {w: i for i, w in enumerate(vocab)}
        self.token_to_word = dict(enumerate(vocab))
        self.num_tokens    = len(vocab)
        self.encoded       = self.encode(word_text)


    def encode(self, words):
        return np.array([self.word_to_token[w] for w in words], dtype=np.int32)

    
    def decode(self, tokens):
        text = " ".join([self.token_to_word[t] for t in tokens])
        return text

    
    def decode_ext(self, tokens):
        text = " ".join([self.token_to_word[t] for t in tokens])
        text = text.replace(' .', '.').replace(' ,', ',')
        snts = text.split('.')
        text = ''
        for s in snts:
            s = s.strip()
            if len(s)<1: 
                continue
            s = s[0].upper() + s[1:]+'. '
            text += s
        return text.strip()

    
    def get_batches(self, n_seqs, n_steps):
        arr = self.encoded
        words_per_batch = n_seqs * n_steps
        n_batches = len(arr)//words_per_batch

        arr = arr[:n_batches * words_per_batch]
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n+n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
