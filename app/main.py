import tensorflow as tf

from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

from app.tokenizer import SequenceTokenizer
from app.model import Encoder, BahdanauAttention, Decoder
from app.utils import CustomUnpickler, load_object, add_sep_tokens, pad_sequence
from app.config import (embedding_dim, units, max_len_encoder, max_len_decoder, checkpoint_dir, tokenizer_encode_path,
                        tokenizer_decode_path)


app = FastAPI()

tokenizer_encode = CustomUnpickler(open(tokenizer_encode_path, 'rb')).load()
tokenizer_decode = CustomUnpickler(open(tokenizer_decode_path, 'rb')).load()

# tokenizer_encode = load_object(tokenizer_encode_path)
# tokenizer_decode = load_object(tokenizer_decode_path)

vocab_size_encode = len(tokenizer_encode.word2index) + 1
vocab_size_decode = len(tokenizer_decode.word2index) + 1

encoder = Encoder(vocab_size_encode, embedding_dim, units, 1)
attention_layer = BahdanauAttention(10)
decoder = Decoder(vocab_size_decode, embedding_dim, units, 1)

optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def predict_v1(words: list):
    words = [add_sep_tokens([letter for letter in word]) for word in words]
    batch_size = len(words)
    result = [['<start>']] * batch_size

    inputs = tokenizer_encode.transform(words)
    inputs = pad_sequence(inputs, max_len=max_len_encoder)
    inputs = tf.convert_to_tensor(inputs)

    hidden = tf.zeros((batch_size, units))
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer_decode.word2index['<start>']] * batch_size, 1)

    for t in range(max_len_decoder):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        prediction_ids = tf.argmax(predictions, axis=1)
        dec_input = tf.expand_dims(prediction_ids, 1)
        for index, id_ in enumerate(prediction_ids.numpy()):
            result[index] = result[index] + [tokenizer_decode.index2word[id_]]

    res = [[item for item in lst if item not in ("<start>", "<end>")] for lst in result]
    return res


class RequestItem(BaseModel):
    id: int
    word: str


class ResponseItem(BaseModel):
    id: int
    phonemes: list


class RequestItemList(BaseModel):
    result: List[RequestItem]


@app.get("/")
def root():
    return "g2p_uk"


@app.post("/predict")
def predict(items: List[RequestItem]):
    res = []
    for item in items:
        res.append({"id": item.id, "phonemes": predict_v1([item.word])})
    return res


@app.post("/predict_list")
def predict_list(items: List[str]):
    res = predict_v1(items)
    return res
