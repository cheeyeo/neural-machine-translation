import argparse
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from model_attention import attention_model, attention_model_new_arch
from model import create_tokenizer, encode_sequences, encode_output
from utils import load_saved_lines, sentence_length, load_tokenizer
from evaluation_utils import predict_attention_sequence, beam_search, cleanup_sentence

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
ap.add_argument("-l", "--limit", type=int, required=False, help="Limit number of test samples")
args = vars(ap.parse_args())

dataset = np.array(load_saved_lines('eng-german-both.pkl'))
test = np.array(load_saved_lines('eng-german-test.pkl'))

# Load tokenizer

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

testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0], padding_type='pre')

model, encoder_model, decoder_model = attention_model_new_arch(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 512)
# model.summary()
model.load_weights(args["model"])

print('[INFO] Evaluating model {}'.format(args["model"]))

if args["limit"] is not None:
  testX = testX[:args["limit"]]
  test = test[:args["limit"]]

print('[INFO] Evaluation Set: {}'.format(len(test)))

evaluate_attention_model(encoder_model, decoder_model, testX, test, ger_tokenizer, ger_vocab_size, ger_length, args["beam"])