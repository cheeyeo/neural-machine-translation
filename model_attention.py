from layers.attention import AttentionLayer
# Using the keras layers within tensorflow due to following issue:
# https://stackoverflow.com/questions/51181754/keras-tensorflow-convlstm2d-object-has-no-attribute-outbound-nodes
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Bidirectional
# from keras.layers import Concatenate
from tensorflow.python.keras.layers import Input, LSTM, Dense, Concatenate, TimeDistributed, Bidirectional, Embedding
from tensorflow.python.keras.models import Model
import numpy as np

def attention_model(src_vocab, target_vocab, src_timesteps, target_timesteps, units):
  encoder_inputs = Input(shape=(src_timesteps,), name='encoder_inputs')

  decoder_inputs = Input(shape=(target_timesteps - 1, target_vocab), name='decoder_inputs')

  embedding = Embedding(src_vocab, units, input_length=src_timesteps, name='enc_embedding')

  encoder_lstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True, name='encoder_lstm'), name='bidirectional_encoder')

  encoder_out, encoder_fwd_state, _, encoder_back_state, _ = encoder_lstm(embedding(encoder_inputs))

  enc_states = [encoder_fwd_state, encoder_back_state]

  # Decoder
  decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='decoder_lstm')
  decoder_out, decoder_state, decoder_back_state = decoder_lstm(decoder_inputs, initial_state=enc_states)

  # Attention
  attn_layer = AttentionLayer(name='attention_layer')
  attn_out, attn_states = attn_layer([encoder_out, decoder_out])

  # concat attention and decoder output
  decoder_output_concat = Concatenate(axis=-1)([decoder_out, attn_out])

  # FC layer
  dense = Dense(target_vocab, activation='softmax')
  time_distributed = TimeDistributed(dense)
  decoder_pred = time_distributed(decoder_output_concat)

  model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Inference models
  # Encoder Inference model
  encoder_inf_inputs = Input(batch_shape=(1, src_timesteps,), name='encoder_inf_inputs')

  encoder_inf_out, encoder_inf_fwd_state, _, encoder_inf_back_state, _ = encoder_lstm(embedding(encoder_inf_inputs))

  encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])


  # # Decoder Inference model
  decoder_inf_inputs = Input(batch_shape=(1, 1, target_vocab), name='decoder_word_inputs')
  encoder_inf_states = Input(batch_shape=(1, src_timesteps, 2*units), name='encoder_inf_states')
  decoder_init_fwd_state = Input(batch_shape=(1, units), name='decoder_fwd_init')
  decoder_init_back_state = Input(batch_shape=(1, units), name='decoder_back_init')

  decoder_inf_out, decoder_inf_fwd_state, decoder_inf_back_state = decoder_lstm(decoder_inf_inputs, initial_state=[decoder_init_fwd_state, decoder_init_back_state])
  
  attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])

  decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])

  decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)

  decoder_model = Model(inputs=[encoder_inf_states, decoder_init_fwd_state, decoder_init_back_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_fwd_state, decoder_inf_back_state])

  return model, encoder_model, decoder_model


if __name__ == "__main__":
  m, _, _ = attention_model(3022, 4747, 5, 12, 256)
  m.summary()