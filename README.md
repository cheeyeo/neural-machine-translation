# Neural Machine Translation

The purpose of this project is to showcase the building of a simple NMT model for language translation in Keras.

## 1. Description

( Describes the problem that is being solved, the source of the data, inputs, and outputs. )

The purpose of the project is to construct a language translation system which automatically translates english to german phrases.

The data is obtained from [http://www.manythings.org/anki/deu-eng.zip](http://www.manythings.org/anki/deu-eng.zip) which is website dedicated to using the Anki flashcard program to learn English as a foreign language.

The data source is a collection of english phrases with its equivalent german translation in a tab delimited file. Each line comprises of the original english phrase and its german translation.

The system will take as input an english phrase and outputs its german translation, based on the data source its trained on. This is a repharsing of the original project goal from 'Deep Learning with NLP' book from which its based on.

The dataset was split into 3 groups: `train, dev, test` sets. Only the train and dev sets are used during the training process. The dev set is used as a validation set after training. The test set is held out until the evaluation process.

Since the dataset comprises of over 190,000 phrases, it is deemed not practical to use it all for this small project. A random selection of 15,000 phrases were selected, of which, 13,000 was used for training; 1000 for dev; 1000 for test.

The data goes through a process of `cleaning` which includes lowercasing; punctuation and non-printable characters removed; removal of non-alphanumeric characters. Future work might include the removal of less frequently used characters to shrink the vocab size as the training corpus increase in size and other data preparation techniques.

The resultant datasets are saved locally in the pickle format.

The code for creating the datasets are in `data-deu.py`. The utility functions for preprocessing the text are in `utils.py`


## 2. Test harness

( Describes how model selection will be performed including the resampling method and model evaluation metrics. )

Model selection will be performed by first training a baseline model from which future models can be developed and evaluated against.

The model evaluation metrics selected are:

* BLEU score

* F1 Score

Once a model is trained and evaluated, its BLEU and F1 scores will be compared to those of the baseline model.

While BLEU scores are not deemed to be the best metric for evaluating NLP systems due to the fact that it doesn't take the context of translations into account, it sufficies for this sample project.

Future work will include looking into and utilizing other NLP evalution metrics.


## 3. Baseline performance

( Describes the baseline model performance (using the test harness) that defines whether a model is skillful or not. )

The code for training the model is in `train.py`. The model configuration and other model-related code can be found in `model.py`

A data generator was used and the model trained using the `fit_generator` function from Keras. This is in preparation for loading more training data in the future, which would result in the overall vocab size to increase exponentially. Loading the entire vocabulary into memory will result in `OOM` errors on the GPU from fitting large tensors during the training process.

The model was trained for 30 epochs with a batch size of 64. A model callback is implented whereby the model gets saved if the validation loss decreases at the end of each epoch.


## 4. Experimental Results

( Presents the experimental results, perhaps testing a suite of models, model configurations, data preparation schemes, and more. Each subsection should have some form of:
4.1 Intent: why run the experiment?
4.2 Expectations: what was the expected outcome of the experiment?
4.3 Methods: what data, models, and configurations are used in the experiment?
4.4 Results: what were the actual results of the experiment?
4.5 Findings: what do the results mean, how do they relate to expectations, what other experiments do they inspire? )

### Intent

The intent of the experiment was to determine if it would be possible to build a neural machine translation system using an Encoder-Decoder architecture which would be able to be trained and translate from enlish to german phrases in an end-to-end manner.

### Expectations

The expectation would be to create a reliable model that could produce a fairly accurate german translation. Given the size of the dataset used, it is not expected for the system to be able to produce a 100% accurate literal translation. Instead, the focus will be on generating a translation that is close/similar in meaning to the actual translation.

### Methods

The baseline model was trained using the training set and the dev set as a validation set. The model structure is based on an `Encoder-Decoder` architecture. The inputs are encoded using a fixed size word embedding, which is passed to a decoder which decodes it word by word to generate a probaility value for each target word in the prediction.

The baseline model uses only 256 units in the LSTM layers.

All the dataset were loaded from and encoded into sequences using the keras `Tokenizer`. Each english sentence is used as input and each german sentence is used as output. Each sequence was padded to the longest sentence in its language. The output is further one-hot encoded using `to_categorical`, as the model will predict the probability of each of the target word in the translation during prediction.

### Results

After training, the model achieved an accuracy of 88% on the train set and approx. 80% on the validation set.

The validation loss graph suggests that the model starts to overfit around epoch 5 with both the training and validation loss diverging further in later epochs.

The model was evaluated against a held out test set, which is run using `evaluate.py`

The computed BLEU scores from evaluation are:

```
BLEU-1: 0.4953
BLEU-2: 0.3695
BLEU-3: 0.2916
BLEU-4: 0.1433
```

The BLEU scores seem reasonable for a small dataset of this size, although the BLEU-1 scores could be higher.

A sample of 10 random translations predicted by the model were chosen from the test set to compare the against the actual translations. It shows that some translations were truncated e.g. `tom wasnt fair` was translated as `tom war nicht` which means `tom was not`

Some translations were wrong e.g. `just check it` was translated as `kontrolliert sie einfach` which means `just control it`

### Findings

The model is fairly accurate on most of the predictions made on the test set and is in line with the expectations set out earlier on dataset of this size. However, it is unable to produce an accurate translation for short phrases in some instances. The results suggest that perhaps the model does not have the capacity to learn and a deeper model might be required.


## 5. Improvements

( Describes experimental results for attempts to improve the performance of the better performing models, such as hyperparameter tuning and ensemble methods. )

An improvement to the baseline model is to add an `Attention` mechanism to the model and build a deeper model to increase the representational capacity of the network. This would allow for larger datasets to be used in training and evaluation.

## 6. Final Model

( Describes the choice of a final model, including configuration and performance. It is a good idea to demonstrate saving and loading the model and demonstrate the ability to make predictions on a holdout dataset. )

[IN PROGRESS]

## 7. Extensions

( Describes areas that were considered but not addressed in the project that could be explored in the future. )

## 8. Resources
* [Deep Learning with NLP](https://machinelearningmastery.com/deep-learning-for-nlp/)

* [Sequence Model course from deep learning specialization](https://www.coursera.org/learn/nlp-sequence-models)
