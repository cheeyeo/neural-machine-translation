from utils import load_clean_lines, sentence_length, load_saved_lines, plot_training
from model import create_tokenizer, baseline_model, create_checkpoint, data_generator
import numpy as np
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train for")
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
model = baseline_model(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 256)

model.summary()
checkpoint = create_checkpoint(model_name='baseline_model.h5')

epochs = args["epochs"]
batch_size = 64

train_steps = len(train) // batch_size
val_steps = len(dev) // batch_size

train_generator = data_generator(train, eng_tokenizer, eng_length, ger_tokenizer, ger_length, ger_vocab_size, batch_size=batch_size)

val_generator = data_generator(dev, eng_tokenizer, eng_length, ger_tokenizer, ger_length, ger_vocab_size, batch_size=batch_size)

H = model.fit_generator(
  train_generator,
  steps_per_epoch=train_steps,
  validation_data=val_generator,
  validation_steps=val_steps,
  epochs=epochs,
  verbose=1,
  callbacks=[checkpoint])

plot_training(H, epochs, plot_path_loss='training_loss_baseline_model.png', plot_path_acc='training_acc_baseline_model.png')
