# Use final model to make predictions on unseen text
import argparse
import numpy as np
# Using the tensorflow version of load_model as custom AttentionLayer was built using TF
from tensorflow.python.keras.models import model_from_json, Model
from tensorflow.python.keras.layers import Input, Concatenate
from layers.attention import AttentionLayer
from keras.utils import to_categorical
from utils import load_saved_lines, sentence_length, clean_lines
from model import create_tokenizer, encode_sequences, encode_output
from model_attention import build_inference_models
from evaluation_utils import predict_attention_sequence, cleanup_sentence

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="Path to model.")
ap.add_argument("-s", "--source", type=str, required=True, help="Sentence to be translated.")
args = vars(ap.parse_args())

# Need to create tokenizers based on entire dataset used in training
# ie. eng-german-both.pkl
dataset = np.array(load_saved_lines('eng-german-both.pkl'))

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = sentence_length(dataset[:, 0])
print('[INFO] English Vocab size: {:d}'.format(eng_vocab_size))
print('[INFO] English Max length: {:d}'.format(eng_length))

ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = sentence_length(dataset[:, 1])
print('[INFO] Ger Vocab size: {:d}'.format(ger_vocab_size))
print('[INFO] Ger Max length: {:d}'.format(ger_length))

# Load model
arch_file = args['model'].split('.')[0] + '.json'
with open('final_model.json', 'rt') as f:
	arch = f.read()

model = model_from_json(arch, custom_objects={'AttentionLayer': AttentionLayer})
model.load_weights(args['model'])
# model.summary()

encoder_model, decoder_model = build_inference_models(eng_length, ger_vocab_size, model)
# encoder_model.summary()
# decoder_model.summary()

# Preprocess input data same as in data-deu.py
source = [args['source']]
source = clean_lines(source)
source = encode_sequences(eng_tokenizer, eng_length, source, padding_type='pre')

seq = encode_sequences(ger_tokenizer, None, ['sos'])
onehot_seq = np.expand_dims(to_categorical(seq, num_classes=ger_vocab_size), 1)

enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(source)
dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state

translation = predict_attention_sequence(decoder_model, enc_outs, dec_fwd_state, dec_back_state, ger_tokenizer, ger_vocab_size, ger_length, onehot_seq)

translation = cleanup_sentence(translation)
print('[INFO] Translation: {}'.format(translation))