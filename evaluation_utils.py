import numpy as np
from keras.utils import to_categorical
from model import encode_sequences

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
    word = word_for_id(dec_ind, target_tokenizer)

    if word == None or word == "eos":
      break

    seq = encode_sequences(target_tokenizer, None, [word])
    onehot_seq = np.expand_dims(to_categorical(seq, num_classes=target_vocab_size), 1)

    predicted_text += word + ' '

  return predicted_text

def beam_search(dec, enc_outs, dec_fwd_state, dec_back_state, target_tokenizer, target_vocab_size, target_length, beam_index):
  # Set none to create encoded seq of single integer
  seq = encode_sequences(target_tokenizer, None, ['sos'])
  in_text = [[seq[0], 0.0]]

  step = 1
  while len(in_text[0][0]) < target_length:
    tempList = []

    for seq in in_text:
      # Use last word in seq?
      recent_word = [seq[0][-1]]
      target = np.expand_dims(to_categorical(recent_word, num_classes=target_vocab_size), 1)
      
      dec_out, attn, dec_fwd_state, dec_back_state = dec.predict(
          [enc_outs, dec_fwd_state, dec_back_state, target])
      preds = dec_out[0][0]

      # top_preds return indices of largest prob values...
      top_preds = np.argsort(preds)[-beam_index:]

      alpha = 0.7
      k = 5.0
      lp = (k + step) ** alpha / (k + 1) ** alpha

      for word in top_preds:
        next_seq, prob = seq[0][:], seq[1]
        next_seq = np.append(next_seq, word)

        # Add length penalty calculation in prob
        prev_length_p = (k + step -1) ** alpha / (k + 1) ** alpha
        prev_length_p = prev_length_p * (step != 1) + (step == 1)
        scores = prob * prev_length_p

        prob = (np.log(preds[word]) + scores) / lp
        tempList.append([next_seq, prob])

    in_text = tempList
    in_text = sorted(in_text, reverse=True, key=lambda l: l[1])
    in_text = in_text[:beam_index]
    step += 1

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