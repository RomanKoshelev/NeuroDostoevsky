import numpy as np

class CharDataset:
    def __init__(self):
        self.text    = None
        self.encoded = []
    
    def load(self, path):
        with open(path, 'r') as f:
            self.text=f.read()
        self.text         = self.text.replace(u'\xa0', u' ').replace(u'\ufeff','') 
        self.vocab        = sorted(set(self.text))
        self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_vocab = dict(enumerate(self.vocab))
        self.encoded      = np.array([self.vocab_to_int[c] for c in self.text], dtype=np.int32)
    
    def get_batches(self, n_seqs, n_steps):
        arr = self.encoded
        characters_per_batch = n_seqs * n_steps
        n_batches = len(arr)//characters_per_batch

        arr = arr[:n_batches * characters_per_batch]
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n+n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
