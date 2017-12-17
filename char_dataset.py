import numpy as np

class Dataset:
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
        '''Создаем генератор, который возвращает пакеты размером
           n_seqs x n_steps из массива arr.

           Аргументы
           ---------
           n_seqs: Batch size, количество последовательностей в пакете
           n_steps: Sequence length, сколько "шагов" делаем в пакете
        '''
        arr = self.encoded
        # Считаем количество символов на пакет и количество пакетов, которое можем сформировать
        characters_per_batch = n_seqs * n_steps
        n_batches = len(arr)//characters_per_batch

        # Сохраняем в массиве только символы, которые позволяют сформировать целое число пакетов
        arr = arr[:n_batches * characters_per_batch]

        # Делаем reshape 1D -> 2D, используя n_seqs как число строк, как на картинке
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            # пакет данных, который будет подаваться на вход сети
            x = arr[:, n:n+n_steps]
            # целевой пакет, с которым будем сравнивать предсказание, получаем сдвиганием "x" на один символ вперед
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
