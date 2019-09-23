from utils import load_doc, to_pairs, save_clean_lines, clean_pairs, load_saved_lines, add_delimiters_to_lines
from model import create_tokenizer
import pickle
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--limit", type=int, required=False, default=15_000, help="Number of sentences to limit to")
args = vars(ap.parse_args())

# limit to first 15,000 sentences from each language
n_sentences = args["limit"]
print('[INFO] Limit dataset to {:d}'.format(n_sentences))

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
dataset[:, 1] = add_delimiters_to_lines(dataset[:, 1])
np.random.shuffle(dataset)
print('[INFO] Total dataset size: {:d}'.format(len(dataset)))

train_size = int(args["limit"] * 0.9)
remainder = args["limit"] - train_size
dev_size = int(remainder/2)
test_size = int(remainder/2)
print("[INFO] Train {:d}, Dev {:d}, Test {:d}".format(train_size, dev_size, test_size))
train, dev, test = dataset[:train_size], dataset[train_size:train_size+dev_size], dataset[train_size+dev_size:train_size+remainder]

save_clean_lines(dataset, 'eng-german-both.pkl')
save_clean_lines(train, 'eng-german-train.pkl')
save_clean_lines(dev, 'eng-german-dev.pkl')
save_clean_lines(test, 'eng-german-test.pkl')
