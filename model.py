import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

def data_generator(lines, eng_tokenizer, eng_length, fr_tokenizer, fr_length, vocab_size, batch_size=64):

  while 1:
    count = 0

    while 1:
      if count >= len(lines):
        count = 0

      input_seq = list()
      output_seq = list()
      for i in range(count, min(len(lines), count+batch_size)):
        eng, fr = lines[i]
        input_seq.append(eng)
        output_seq.append(fr)
      input_seq = encode_sequences(eng_tokenizer, eng_length, input_seq)
      output_seq = encode_sequences(fr_tokenizer, fr_length, output_seq)
      output_seq = encode_output(output_seq, vocab_size)

      count = count + batch_size

      input_seq = np.array(input_seq)
      output_seq = np.array(output_seq)
      output_seq = output_seq.reshape((output_seq.shape[0], output_seq.shape[1],vocab_size))
      yield [input_seq, output_seq]


def create_checkpoint(model_name='model.h5'):
  return ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

def create_tokenizer(lines):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

def encode_sequences(tokenizer, length, lines):
  X = tokenizer.texts_to_sequences(lines)
  X = pad_sequences(X, maxlen=length, padding='post')
  return X

def encode_output(sequences, vocab_size):
  ylist = list()
  for seq in sequences:
    encoded = to_categorical(seq, num_classes=vocab_size)
    ylist.append(encoded)
  y = np.array(ylist)
  y = y.reshape((sequences.shape[0], sequences.shape[1], vocab_size))
  return y

def baseline_model(src_vocab, target_vocab, src_timesteps, target_timesteps, units):
  model = Sequential()
  model.add(Embedding(src_vocab, units, input_length=src_timesteps, mask_zero=True))
  model.add(LSTM(units))
  model.add(RepeatVector(target_timesteps))
  # decoder model
  model.add(LSTM(units, return_sequences=True))
  model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
