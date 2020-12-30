from app.utils import flatten


class SequenceTokenizer:

    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.oov_token = '<UNK>'
        self.oov_token_index = 0

    def fit(self, sequence):
        self.index2word = dict(enumerate([self.oov_token] + sorted(set(flatten(sequence))), 1))
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.oov_token_index = self.word2index.get(self.oov_token)
        return self

    def transform(self, X):
        res = []
        for line in X:
            res.append([self.word2index.get(item, self.oov_token_index) for item in line])
        return res
