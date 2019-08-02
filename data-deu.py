from utils import load_doc, to_pairs, save_clean_lines, clean_pairs, load_saved_lines
from model import create_tokenizer
import pickle
import numpy as np

# limit to first 15,000 sentences from each language
n_sentences = 15_000

filename = 'data/deu.txt'

doc = load_doc(filename)
# print(len(doc))

pairs = to_pairs(doc)
print(len(pairs))
print(pairs[0])

cleaned = clean_pairs(pairs)

for i in range(10):
  print('[{}] => [{}]'.format(cleaned[i, 0], cleaned[i, 1]))

# Split the data
dataset = cleaned[:n_sentences, :]
np.random.shuffle(dataset)

train, dev, test = dataset[:13000], dataset[13000:14000], dataset[14000:15000]

save_clean_lines(dataset, 'eng-german-both.pkl')
save_clean_lines(train, 'eng-german-train.pkl')
save_clean_lines(dev, 'eng-german-dev.pkl')
save_clean_lines(test, 'eng-german-test.pkl')
