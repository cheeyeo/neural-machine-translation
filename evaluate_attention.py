from model import create_tokenizer, encode_sequences, encode_output
import argparse
from utils import load_saved_lines, sentence_length
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import classification_report
from model_attention import attention_model
from keras.utils import to_categorical

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def cleanup_sentence(sentence):
  index = sentence.find('startseq ')
  if index > -1:
    sentence = sentence[len('startseq '):]
  index = sentence.find(' endseq')
  if index > -1:
    sentence = sentence[:index]
  return sentence

def predict_attention_sequence(decoder_model, enc_outs, dec_state, target_tokenizer, target_vocab_size, target_length, onehot_seq):
  predicted_text = ''

  for i in range(target_length):
    dec_out, attention, dec_state = decoder_model.predict(
              [enc_outs, dec_state, onehot_seq])

    dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
    word = word_for_id(dec_ind, target_tokenizer)

    if word == None or word == "endseq":
      break

    seq = encode_sequences(ger_tokenizer, None, [word])
    onehot_seq = np.expand_dims(to_categorical(seq, num_classes=target_vocab_size), 1)

    predicted_text += word + ' '

  return predicted_text

def evaluate_attention_model(enc, dec, sources, raw_dataset, target_tokenizer, target_vocab_size, target_length):
  actual, predicted = list(), list()

  for i, source in enumerate(sources):
    seq = encode_sequences(target_tokenizer, None, ['startseq'])
    onehot_seq = np.expand_dims(to_categorical(seq, num_classes=target_vocab_size), 1)

    source = source.reshape((1, source.shape[0]))
    enc_outs, enc_fwd_state, enc_back_state = enc.predict(source)

    # Joins the fwd and back states to be of shape (2, lstm units) to pass to the decoder.
    dec_state = np.concatenate([enc_fwd_state, enc_back_state], axis=0)
    
    translation = predict_attention_sequence(dec, enc_outs, dec_state, target_tokenizer, target_vocab_size, target_length, onehot_seq)

    raw_src, raw_target = raw_dataset[i]
    raw_target = cleanup_sentence(raw_target)

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

# Load ger tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = sentence_length(dataset[:, 1])
print('[INFO] Ger Vocab size: {:d}'.format(ger_vocab_size))
print('[INFO] Ger Max length: {:d}'.format(ger_length))

testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

model, encoder_model, decoder_model = attention_model(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 256)
# model.summary()
model.load_weights(args["model"])

print('[INFO] Evaluating model {}'.format(args["model"]))

evaluate_attention_model(encoder_model, decoder_model, testX, test, ger_tokenizer, ger_vocab_size, ger_length)
