from dataset import TextDataset

class MathDataset(TextDataset):
    def _read_text(self, path, voc, min_len, max_len):
        with open(path, 'r') as f:
            text = f.read()
            text = text.lower()
        tokenized_text = []

        for sentence in text.split('\n'):
            words = list(sentence)
            if min_len <= len(words) <= max_len:
                tokens = voc.to_tokens(words)
                tokenized_text.append(tokens)
        return tokenized_text
    