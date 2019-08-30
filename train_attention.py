# Training file for Attention model
from model_attention import attention_model, attention_model_new_arch
from model import create_checkpoint, create_tokenizer, encode_sequences, encode_output
import argparse
import numpy as np
from utils import load_saved_lines, sentence_length, plot_training
from keras.utils import plot_model

def data_generator(lines, eng_tokenizer, eng_length, fr_tokenizer, fr_length, src_vocab_size, vocab_size, batch_size=64):

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
      input_seq = encode_sequences(eng_tokenizer, eng_length, input_seq, padding_type='pre')
      output_seq = encode_sequences(fr_tokenizer, fr_length, output_seq)
      output_seq = encode_output(output_seq, vocab_size)

      count = count + batch_size

      input_seq = np.array(input_seq)
      output_seq = np.array(output_seq)
      decoder_inputs = output_seq[:, :-1, :]
      decoder_outputs = output_seq[:, 1:, :]
      yield [[input_seq, decoder_inputs], decoder_outputs]

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train for")
ap.add_argument("-b", "--batch", type=int, required=False, default=64, help="Batch size")
ap.add_argument("-m", "--model", type=str, required=False, default="attention_model.h5", help="Model name to save to.")
args = vars(ap.parse_args())

dataset = np.array(load_saved_lines('eng-german-both.pkl'))
train = np.array(load_saved_lines('eng-german-train.pkl'))
for i in range(5):
  print(train[i])
dev = np.array(load_saved_lines('eng-german-dev.pkl'))
for i in range(5):
  print(dev[i])
print('[INFO] Training set size: {:d}'.format(len(train)))
print('[INFO] Dev set size: {:d}'.format(len(dev)))

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

print('[INFO] Defining model...')
model, encoder_model, decoder_model = attention_model_new_arch(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 512)

plot_model(model, to_file='artifacts/attention_model_new_arch.png', show_shapes=True)
plot_model(encoder_model, to_file='artifacts/encoder_model_new_arch.png', show_shapes=True)
plot_model(decoder_model, to_file='artifacts/decoder_model_new_arch.png', show_shapes=True)

model.summary()
checkpoint = create_checkpoint(model_name=args["model"])

epochs = args["epochs"]
batch_size = args["batch"]

train_steps = len(train) // batch_size
val_steps = len(dev) // batch_size

train_generator = data_generator(train, eng_tokenizer, eng_length, ger_tokenizer, ger_length, eng_vocab_size, ger_vocab_size, batch_size=batch_size)

val_generator = data_generator(dev, eng_tokenizer, eng_length, ger_tokenizer, ger_length, eng_vocab_size, ger_vocab_size, batch_size=batch_size)

H = model.fit_generator(
  train_generator,
  steps_per_epoch=train_steps,
  validation_data=val_generator,
  validation_steps=val_steps,
  epochs=epochs,
  verbose=1,
  callbacks=[checkpoint])

plot_training(H, epochs, plot_path_loss='training_loss_attention_model_new_arch.png', plot_path_acc='training_acc_attention_model_new_arch.png')
