from model import create_tokenizer, encode_sequences, encode_output
import argparse
from utils import load_saved_lines, sentence_length, load_tokenizer
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import classification_report
from model_attention import attention_model, attention_model_new_arch
from keras.utils import to_categorical
from keras.models import model_from_json
from math import log

def id_for_word(word, tokenizer):
  for w,index in tokenizer.word_index.items():
    if w == word:
      return index
  return None

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def cleanup_sentence(sentence):
  index = sentence.find('sos ')
  if index > -1:
    sentence = sentence[len('sos '):]
  index = sentence.find(' eos')
  if index > -1:
    sentence = sentence[:index]
  return sentence

def predict_attention_sequence(decoder_model, enc_outs, dec_fwd_state, dec_back_state, target_tokenizer, target_vocab_size, target_length, onehot_seq):
  predicted_text = ''

  for i in range(target_length):
    dec_out, attention, dec_fwd_state, dec_back_state = decoder_model.predict(
              [enc_outs, dec_fwd_state, dec_back_state, onehot_seq], verbose=0)

    dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
    # print(dec_ind)
    word = word_for_id(dec_ind, target_tokenizer)
    # print('WORD: ', word)

    if word == None or word == "eos":
      break

    seq = encode_sequences(ger_tokenizer, None, [word])
    onehot_seq = np.expand_dims(to_categorical(seq, num_classes=target_vocab_size), 1)

    predicted_text += word + ' '

  return predicted_text

def beam_search(dec, enc_outs, dec_fwd_state, dec_back_state, target_tokenizer, target_vocab_size, target_length, beam_index):
  # Set none to create encoded seq of single integer
  seq = encode_sequences(target_tokenizer, None, ['sos'])
  in_text = [[seq[0], 1.0]]

  step = 0
  while len(in_text[0][0]) < target_length:
    # print('CURRENT STEP: {}'.format(step))

    tempList = []

    # print('SORTED IN_TEXT: ', in_text)

    for seq in in_text:
      # Use last word in seq?
      recent_word = [seq[0][-1]]
      # recent_word = seq[0]
      # print('RECENT WORD: ', recent_word)
      target = np.expand_dims(to_categorical(recent_word, num_classes=target_vocab_size), 1)
      
      dec_out, _, dec_fwd_state, dec_back_state = dec.predict(
          [enc_outs, dec_fwd_state, dec_back_state, target])
      # preds = dec_out[0][0]

      # Log softmax output
      logits = dec_out[0][0]
      preds = logits - np.log(np.sum(logits, axis=-1))

      # top_preds return indices of largest prob values...
      top_preds = np.argsort(preds)[-beam_index:]

      # for x in top_preds:
      #   print('ID: {}, WORD: {}, PRED: {}'.format(x, word_for_id(x, target_tokenizer), np.log(preds[x])))

      for word in top_preds:
        next_seq, prob = seq[0][:], seq[1]
        next_seq = np.append(next_seq, word)
        # Add length penalty calculation in prob
        alpha = 0.7
        prev_length_p = (5 + step -1) ** alpha / (5 + 1) ** alpha
        prev_length_p = prev_length_p * (step != 1) + (step == 1)
        scores = prob * prev_length_p
        # print('Scores: ', scores)
        lp = (5 + step) ** alpha / (5 + 1) ** alpha
        prob = (np.log(preds[word]) + scores) / lp
        tempList.append([next_seq, prob])

    # print('TEMPLIST: ', tempList)
    # print()

    in_text = tempList
    # TODO: Prune entries by removing hypotheses with low scores below certain threshold?
    in_text = sorted(in_text, reverse=True, key=lambda l: l[1])
    # print('TEMPLIST AFTER SORT: ', in_text)
    in_text = in_text[:beam_index]
    step += 1
  
  # print('AFTER WHILE LOOP, IN_TEXT: ', in_text)
  in_text = in_text[0][0]
  final_caption_raw = [word_for_id(i, target_tokenizer) for i in in_text]
  final_caption = []
  for word in final_caption_raw:
    if word=='eos' or word==None:
      break
    else:
      final_caption.append(word)
  final_caption.append('eos')
  return ' '.join(final_caption)


def evaluate_attention_model(enc, dec, sources, raw_dataset, target_tokenizer, target_vocab_size, target_length, beam_index):
  actual, predicted = list(), list()

  if beam_index is not None:
    print('[INFO] Evaluating with beam search of width: {:d}'.format(beam_index))
    print()

  for i, source in enumerate(sources):
    print('[INFO] Translating idx: ', i)
    seq = encode_sequences(target_tokenizer, None, ['sos'])
    onehot_seq = np.expand_dims(to_categorical(seq, num_classes=target_vocab_size), 1)

    source = source.reshape((1, source.shape[0]))
    enc_outs, enc_fwd_state, enc_back_state = enc.predict(source)

    dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state

    if beam_index is None:
      translation = predict_attention_sequence(dec, enc_outs, dec_fwd_state, dec_back_state, target_tokenizer, target_vocab_size, target_length, onehot_seq)
    else:
      translation = beam_search(dec, enc_outs, dec_fwd_state, dec_back_state, target_tokenizer, target_vocab_size, target_length, beam_index)

    translation = cleanup_sentence(translation)

    raw_src, raw_target = raw_dataset[i]
    raw_target = cleanup_sentence(raw_target)

    print('[INFO] SRC={}, TARGET={}, PREDICTED={}'.format(raw_src, raw_target, translation))
    print()

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
ap.add_argument("-k", "--beam", type=int, required=False, help="Beam width for beam search")
args = vars(ap.parse_args())

dataset = np.array(load_saved_lines('eng-german-both.pkl'))
test = np.array(load_saved_lines('eng-german-test.pkl'))

# Load tokenizer

# Load eng tokenizer
# eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_tokenizer = load_tokenizer('eng_tokenizer.pkl')
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = sentence_length(dataset[:, 0])
print('[INFO] English Vocab size: {:d}'.format(eng_vocab_size))
print('[INFO] English Max length: {:d}'.format(eng_length))

# Load ger tokenizer
# ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_tokenizer = load_tokenizer('ger_tokenizer.pkl')
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = sentence_length(dataset[:, 1])
print('[INFO] Ger Vocab size: {:d}'.format(ger_vocab_size))
print('[INFO] Ger Max length: {:d}'.format(ger_length))

testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0], padding_type='pre')

model, encoder_model, decoder_model = attention_model_new_arch(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 512)
# model.summary()
model.load_weights(args["model"])

print('[INFO] Evaluating model {}'.format(args["model"]))

print('[INFO] Evaluation Set: {}'.format(len(test)))

evaluate_attention_model(encoder_model, decoder_model, testX, test, ger_tokenizer, ger_vocab_size, 6, args["beam"])