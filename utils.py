import string
import re
import pickle
from unicodedata import normalize
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def plot_training(H, N, plot_path_loss="training_loss.png", plot_path_acc="training_acc.png"):
  plt.style.use("ggplot")
  plt.figure()

  plt.title("Training Loss")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss")
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.legend(loc="lower left")
  plt.savefig(plot_path_loss)

  plt.clf()

  plt.title("Training Acc")
  plt.xlabel("Epoch #")
  plt.ylabel("Acc")
  plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
  plt.legend(loc="lower left")
  plt.savefig(plot_path_acc)


def clean_pairs(lines):
  cleaned = list()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  for pair in lines:
    clean_pair = list()
    for line in pair:
      line = normalize('NFD', line).encode('ascii', 'ignore')
      line = line.decode('UTF-8')
      line = line.split()
      line = [w.lower() for w in line]
      line = [re_punc.sub('', w) for w in line]
      line = [re_print.sub('', w) for w in line]
      line = [w for w in line if w.isalpha()]
      clean_pair.append(' '.join(line))
    cleaned.append(clean_pair)
  return np.array(cleaned)

def clean_lines(lines):
  cleaned = list()
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  # translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)

  for line in lines:
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    line = line.split()
    line = [w.lower() for w in line]
    # Remove punctuation
    line = [w.translate(table) for w in line]

    # Remove single characters left from aprostrophe removal etc...
    line = [w for w in line if len(w) > 1]

    # Remove non-printable characters
    line = [re_print.sub('', w) for w in line]
    line = [w for w in line if w.isalpha()]
    cleaned.append(' '.join(line))

  return cleaned

def save_clean_lines(sentences, filename):
  pickle.dump(sentences, open(filename, 'wb'))
  print('[INFO] Saved clean lines: {}'.format(filename))


def save_tokenizer(tokenizer, filename):
  pickle.dump(tokenizer, open(filename, 'wb'))
  print('[INFO] Saved tokenizer: {}'.format(filename))

def load_doc(filename):
  """
  Opens up a file with utf8 encoding
  """
  with open(filename, mode='rt', encoding='utf-8') as f:
    text = f.read()
    return text

def to_pairs(doc):
  lines = doc.strip().split('\n')
  pairs = [line.split('\t') for line in lines]
  return pairs

def to_sentences(doc):
  """
  Turns a binary doc file object into a list
  """
  return doc.strip().split('\n')

def sentence_length(sentences):
  """
  Return max lengths for list of sentences
  """
  lengths = [len(s.split()) for s in sentences]
  return max(lengths)

def load_saved_lines(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)

def load_clean_lines(filename):
  doc = load_saved_lines(filename)
  lines = list()
  for line in doc:
    new_line = 'startseq ' + line + ' endseq'
    lines.append(new_line)
  return lines

def add_delimiters_to_lines(lines):
  new_lines = list()
  for line in lines:
    new_line = 'sos ' + line + ' eos'
    new_lines.append(new_line)
  return new_lines

def to_vocab(lines):
  vocab = Counter()
  for line in lines:
    tokens = line.split()
    vocab.update(tokens)
  return vocab

def trim_vocab(vocab, min_occurences):
  tokens = [k for k, c in vocab.items() if c >= min_occurences]
  return set(tokens)

def update_dataset(lines, vocab):
  new_lines = list()
  for line in lines:
    new_tokens = list()

    for token in line.split():
      if token in vocab:
        new_tokens.append(token)
      else:
        new_tokens.append('unk')
    new_line = ' '.join(new_tokens)
    new_lines.append(new_line)

  return new_lines

if __name__ == '__main__':
  # lines = ["Please rise, then, for this minute' s silence."]
  # res = clean_lines(lines)
  # print(res)

  lines = ['startseq resumption of the session endseq', 'startseq declare resumed the session of the european parliament adjourned on friday december and would like once again to wish you happy new year in the hope that you enjoyed pleasant festive period endseq']

  vocab = to_vocab(lines)
  print(vocab)

  res = update_dataset(lines, vocab)
  print(res)
