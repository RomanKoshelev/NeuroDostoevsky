from vocabulary import Vocabulary

class MathVocab(Vocabulary):
    def _update(self, fdist, path):
        with open(path, 'r') as f:
            text = f.read()
            text = text.lower()

        for sentence in text.split('\n'):
            for word in sentence:
                fdist[word]+=1
        return fdist
