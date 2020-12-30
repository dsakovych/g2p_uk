import pickle
import tensorflow as tf


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'SequenceTokenizer':
            from app.tokenizer import SequenceTokenizer
            return SequenceTokenizer
        return super().find_class(module, name)


def load_object(filename):
    with open(filename, 'rb') as inpt:
        obj = pickle.load(inpt)
    return obj


def flatten(array):
    for item in array:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def add_sep_tokens(x):
    return ['<start>'] + x + ['<end>']


def pad_sequence(x, max_len=None):
    return tf.keras.preprocessing.sequence.pad_sequences(x,
                                                         padding='post',
                                                         maxlen=max_len)


