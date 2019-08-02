from layers.attention import AttentionLayer
# Using the keras layers within tensorflow due to following issue:
# https://stackoverflow.com/questions/51181754/keras-tensorflow-convlstm2d-object-has-no-attribute-outbound-nodes
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Bidirectional
# from keras.layers import Concatenate
from tensorflow.python.keras.layers import Input, LSTM, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.models import Model

def attention_model(src_vocab, target_vocab, src_timesteps, target_timesteps, units):
  encoder_inputs = Input(shape=(src_timesteps, src_vocab), name='encoder_inputs')
  decoder_inputs = Input(shape=(target_timesteps-1, target_vocab), name='decoder_inputs')

  encoder_lstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True, name='encoder_gru'), name='bidirectional_encoder')
  encoder_out, encoder_fwd_state, encoder_fwd_c, encoder_back_state, encoder_back_c = encoder_lstm(encoder_inputs)
  encoder_fwd_state = Concatenate()([encoder_fwd_state, encoder_back_state])
  encoder_back_state = Concatenate()([encoder_fwd_c, encoder_back_c])

  # decoder
  decoder_lstm = LSTM(units*2, return_sequences=True, return_state=True, name='decoder_lstm')
  decoder_out, decoder_state, _ = decoder_lstm(decoder_inputs, initial_state=[encoder_fwd_state, encoder_back_state])

  # attention
  # TODO: Issue with tensorflow code in the layer itself!!
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

  return model
