import os

embedding_dim = 256
units = 1024

max_len_encoder = 35
max_len_decoder = 33

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, 'data')

checkpoint_dir = os.path.join(DATA_DIR, 'tf_checkpoints', 'phones_with_stress')
tokenizer_encode_path = os.path.join(DATA_DIR, 'tokenizer_encode.pkl')
tokenizer_decode_path = os.path.join(DATA_DIR, 'tokenizer_decode.pkl')
