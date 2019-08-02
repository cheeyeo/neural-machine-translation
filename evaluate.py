from model import create_tokenizer, encode_sequences
import argparse
from utils import load_saved_lines, sentence_length
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def predict_sequence(model, tokenizer, source):
  prediction = model.predict(source, verbose=0)[0]
  integers = [np.argmax(vector) for vector in prediction]

  target = list()
  for i in integers:
    word = word_for_id(i, tokenizer)
    if word is None:
      break
    target.append(word)

  return ' '.join(target)


def evaluate_model(model, sources, raw_dataset, fr_tokenizer, beam_index=3):
  actual, predicted = list(), list()
  for i, source in enumerate(sources):
    source = source.reshape((1, source.shape[0]))
    translation = predict_sequence(model, fr_tokenizer, source)

    raw_src, raw_target = raw_dataset[i]
    if i < 10:
      print('[INFO] SRC={}, TARGET={}, PREDICTED={}'.format(raw_src, raw_target, translation))

    actual.append([raw_target.split()])
    predicted.append(translation.split())
  bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
  bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
  bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
  bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
  print('BLEU-1: {:.4f}'.format(bleu1))
  print('BLEU-2: {:.4f}'.format(bleu2))
  print('BLEU-3: {:.4f}'.format(bleu3))
  print('BLEU-4: {:.4f}'.format(bleu4))


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="Path to model file for evaluation")
ap.add_argument("-k", "--beam", type=int, required=False, default=3, help="Beam width for beam search")
args = vars(ap.parse_args())

dataset = np.array(load_saved_lines('eng-german-both.pkl'))
test = np.array(load_saved_lines('eng-german-test.pkl'))

# Load eng tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = sentence_length(dataset[:, 0])
print('[INFO] English Vocab size: {:d}'.format(eng_vocab_size))
print('[INFO] English Max length: {:d}'.format(eng_length))

# Load fr tokenizer
fr_tokenizer = create_tokenizer(dataset[:, 1])
fr_vocab_size = len(fr_tokenizer.word_index) + 1
fr_length = sentence_length(dataset[:, 1])
print('[INFO] Ger Vocab size: {:d}'.format(fr_vocab_size))
print('[INFO] Ger Max length: {:d}'.format(fr_length))

testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

model = load_model(args["model"])

print('[INFO] Evaluating model {} with beam width {:d}'.format(args["model"], args["beam"]))

evaluate_model(model, testX, test, fr_tokenizer)
