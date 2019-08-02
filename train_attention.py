# Training file for Attention model
from model_attention import attention_model
from model import create_checkpoint
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train for")
args = vars(ap.parse_args())

print('[INFO] Defining model...')
# Hardcode vocab and sentence length for now as dataset needs to be processed differently....
eng_vocab_size = 3022
eng_length = 5
ger_vocab_size = 4745
ger_length = 10
model = attention_model(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 512)

model.summary()
checkpoint = create_checkpoint(model_name='baseline_model.h5')

epochs = args["epochs"]
batch_size = 64
