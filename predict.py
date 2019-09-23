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
arch_file = args["model"].split(".")[0] + '.json'
with open('final_model.json', 'rt') as f:
	arch = f.read()

model = model_from_json(arch, custom_objects={'AttentionLayer': AttentionLayer})
model.load_weights(args["model"])

# model.summary()
# TODO: Infer nos of units from model arch!
units = 512

# Encoder inference model
encoder_inf_inputs = Input(batch_shape=(1, eng_length,), name='encoder_inf_inputs')
encoder_lstm = model.get_layer('bidirectional_encoder')
embedding = model.get_layer('enc_embedding')
encoder_inf_out, encoder_inf_fwd_h, encoder_inf_fwd_c, encoder_inf_back_h, encoder_inf_back_c = encoder_lstm(embedding(encoder_inf_inputs))

encoder_inf_h = Concatenate()([encoder_inf_fwd_h, encoder_inf_back_h])
encoder_inf_c = Concatenate()([encoder_inf_fwd_c, encoder_inf_back_c])
encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_h, encoder_inf_c])
# encoder_model.summary()

# Decoder inference model
encoder_inf_states = Input(batch_shape=(1, eng_length, 2*units), name='encoder_inf_states')

decoder_inf_inputs = Input(batch_shape=(1, 1, ger_vocab_size), name='decoder_word_inputs')

decoder_init_fwd_state = Input(batch_shape=(1, units*2), name='decoder_fwd_init')
decoder_init_back_state = Input(batch_shape=(1, units*2), name='decoder_back_init')
decoder_states_inputs = [decoder_init_fwd_state, decoder_init_back_state]

decoder_lstm = model.get_layer('decoder_lstm')
decoder_inf_out, decoder_inf_fwd_state, decoder_inf_back_state = decoder_lstm(decoder_inf_inputs, initial_state=decoder_states_inputs)

attn_layer = model.get_layer('attention_layer')
attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])

decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])

decoder_dense = model.layers[-1]
decoder_inf_pred = decoder_dense(decoder_inf_concat)

decoder_model = Model(inputs=[encoder_inf_states, decoder_init_fwd_state, decoder_init_back_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_fwd_state, decoder_inf_back_state])

# decoder_model.summary()

# Preprocess input data same as in data-deu.py
source = [args["source"]]
source = clean_lines(source)
source = encode_sequences(eng_tokenizer, eng_length, source, padding_type='pre')

seq = encode_sequences(ger_tokenizer, None, ['sos'])
onehot_seq = np.expand_dims(to_categorical(seq, num_classes=ger_vocab_size), 1)

enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(source)
dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state

translation = predict_attention_sequence(decoder_model, enc_outs, dec_fwd_state, dec_back_state, ger_tokenizer, ger_vocab_size, ger_length, onehot_seq)

translation = cleanup_sentence(translation)
print('[INFO] Translation: {}'.format(translation))