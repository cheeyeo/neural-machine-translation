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

[training_curve]: artifacts/training_acc_baseline_model.png
[loss_curve]: artifacts/training_loss_baseline_model.png

After training, the model achieved an accuracy of 88% on the train set and approx. 80% on the validation set.

![baseline model training curve][training_curve]

The validation loss graph suggests that the model starts to overfit around epoch 5 with both the training and validation loss diverging further in later epochs.

![baseline model validation loss][loss_curve]

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


## 5.1 Adding an attention layer
An improvement to the baseline model is to add an `Attention` mechanism to the model and build a deeper model to increase the representational capacity of the network. This would allow for larger datasets to be used in training and evaluation.

The Attention mechanism is based on a weighted Attention mechanism published by 
Bahdanau. This is implemented as a custom keras layer.

The new model architecture was trained for 30 epochs with a batch size of 64.

The validation loss shows a better accuracy with a final score of `0.94348` compared to the baseline model's loss values at between 1.0-1.5. The accuracy curves also indicate an improvement on the overall training accuracy score, with the training accuracy rising to 0.95.

However, the gaps between the loss curves indicate overfitting around epoch 10 onwards, with the validation loss rising after 10 epochs and rising above 1.0 from epochs 25 onwards.

The BLEU scores for evaluation are as follows:
```
BLEU-1: 0.5894
BLEU-2: 0.4667
BLEU-3: 0.3998
BLEU-4: 0.2440
```

Compared to the BLEU scores for the baseline model, it can be seen that adding an attention mechanism has increased the BLEU-1 to BLEU-4 scores.

## 5.2 Refining the attention layer

The attention mechanism is implemented as a custom Keras layer which does not support masking, which can be turned on in the word Embedding layer. 

By setting `mask_zero=True` in the Embedding layer, we are informing the model that some of the values are padding and should be ignored.

I implemented the `compute_mask` function in the attention layer to return `None` so as to pass the mask down to further layers. 

The validation loss dropped to `0.89285` after training for 30 epochs with a batch size of 64.

The loss curves also indicate that the validation loss has reduced to below 1.0 level across all epochs although the gap between the loss and validation loss is still high, indicating overfitting.

Running model evaluation after the changes show an increase in the BLEU scores compared to 5.1 above:

```
BLEU-1: 0.6249
BLEU-2: 0.5145
BLEU-3: 0.4552
BLEU-4: 0.3046
```

## 5.3 Addressing Variance / Overfitting - More training data

Increase the training data set size in increments of 5000 and observe the effects on the validation loss and when it starts to overfit...

Each increase in data set is trained for 30 epochs with batch size of 32.

| Dataset size | Validation Loss | Start to overfit at epoch |
| 20000				 | 0.84723				 | 8								         |
| 25000				 | 0.82068			   | 5							           |
| 30000				 | 0.83362		     | 4							           |

From the above, we can see that increasing the dataset to 25,000 gives the lowest validation loss. The epoch at which the model starts to overfit also matches with the original training dataset size.

[attention_loss_curve]: artifacts/training_loss_attention_model_data_25.png

![attention model loss curve][attention_loss_curve]

The plot of the validation loss still indicates that the model is overfitting from the gap between the training and validation losses and also the sudden increase in loss from epoch 5 onwards.

The BLEU scores on evaluation are:
```
BLEU-1: 0.6196
BLEU-2: 0.4945
BLEU-3: 0.4304
BLEU-4: 0.2860
```

All the BLEU scores show an increase apart from the BLEU-3 scores which might indicate an issue with shorter phrases that would require further investigation.

## 5.4 Addressing Variance - Dropout Regularization

[attention_model_dropout]: artifacts/attention_model_dropout.png
[training_loss_dropout]: artifacts/training_loss_attention_model_new_arch_dropout.png

![attention model with dropout][attention_model_dropout]

Experiment with adding dropout layers in the following scenarios:

* Adding dropout layer after LSTM encoder

* Adding dropout layer after embedding layer

* Adding dropout layer after embedding and FC layers

The validation loss increased to `0.86515` with dropout applied to after the LSTM encoder layer.

The validation loss decreased to `0.844667` with dropout applied to after the embedding layer which is still higher than in section 5.4

Applying dropout to after both the embedding and FC layers achieved a much lower validation loss of `0.80030`

![Validation loss of attention model with dropout][training_loss_dropout]

The validation loss curve still shows inflections of increase in the loss values rather than a gradual decline. This suggests that more regularization is needed.

The BLEU scores do show an improvement during evaluation:
  ```
  BLEU-1: 0.6251
  BLEU-2: 0.5034
  BLEU-3: 0.4422
  BLEU-4: 0.2973
  ```

The use of dropout will be applied to the model and further regularization techniques will be used to further reduce the validation loss.


## 5.5 Addressing Variance - Architecture Search

[attention_model_new_arch]: artifacts/attention_model_new_arch.png
[training_loss_new_arch]: artifacts/training_loss_attention_model_new_arch.png

![Attention model with dropout and double decoder units][attention_model_new_arch]

The number of decoder LSTM units are doubled from 512 to 1024 to match the number of bi-directional LSTM units. The time distributed layer is also removed in the FC layers.

![Training loss with dropout and larger decoder layer][training_loss_new_arch]

The validation loss increased to `0.84758` compared to 5.4.

The loss curves show that the validation loss still has inflection points where the loss values increase.

This indicates that further regularization is needed as the model is overfit.


## 5.6 Addressing Variance - Early Stopping

Based on the observations in the training curves, the model starts to overfit after certain epochs. By training the model long enough to learn the mapping from inputs to outputs and not too long to overfit on the training data, we aim to derive a better model which generalizes better.

Early stopping is implemented as a callback which monitors the validation loss and stops the training process once the loss stops decreasing. A `patience` threshold of 2 is set to allow the model to continue training for an additional 2 epochs in case the loss is at a plateau and a lower value can be derived.

The model checkpoint is still used as a callback which is invoked after early stopping. It saves the model's weights with the lowest validation loss once training stops.

Given the model's configuration to date, adopting early stopping helps to shorten the training time but the validation loss is still unchanged as in 5.5.

## 5.7 Addressing Variance - Weight constraints

Experiment with adding weight constraints to the LSTM layers.

Weight constraints work well with other regularization techniques such as dropout.

Weight constraints penalizes large weights during update and imposes a penalty on the loss function, which in turn causes the weights to be small and helps prevent overfitting.

The weight constraints are applied to the input connections using `kernel_constraint` or recurrent connections `recurrent_constraint` for the LSTM layers. 

We will consider only two types of norms, `unit_norm` and `max_norm`

`unit_norm` forces weights to have a magnitude of 1.0.

`max_norm` forces weights to have magnitude at or below a given value.

The weight constraints are applied to both the encoder and decoder LSTM layers.

Adding `kernel_constraint` to `unit_norm` to the encoder achieves a validation loss of `0.82301`

Adding `kernel_constraint` to `unit_norm` to the encoder and decoder achieves a higher validation loss of `0.9813`

[training_loss_new_arch_unitnorm]: artifacts/training_loss_attention_model_new_arch_unitnorm.png

[training_acc_new_arch_unitnorm]: artifacts/training_acc_attention_model_new_arch_unitnorm.png

![Training loss with unit_norm applied to LSTM Layers][training_loss_new_arch_unitnorm]

![Training acc with unit_norm applied to LSTM Layers][training_acc_new_arch_unitnorm]

In both instances, the validation loss curve above still show inflection points where the loss values increase. This is confirmed in the training accuracy curves where the training accuracy starts to decrease after epoch `7.5`

Based on the above, adding weight constraints to the encoder only seem to derive lower validation loss.

Adding `max_norm` to `kernel_constraint` to the encoder only achieves a lower loss of `0.80447`

The training log below shows that early stopping kicked in, stopping the training process at epoch 7

```
702/703 [============================>.] - ETA: 0s - loss: 1.9844 - acc: 0.7183         
Epoch 00001: val_loss improved from inf to 1.44963, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 196s 278ms/step - loss: 1.9835 - acc: 0.7185 - val_loss: 1.4496 - val_acc: 0.7862
Epoch 2/30
702/703 [============================>.] - ETA: 0s - loss: 1.2442 - acc: 0.8025  
Epoch 00002: val_loss improved from 1.44963 to 1.11857, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 191s 272ms/step - loss: 1.2439 - acc: 0.8025 - val_loss: 1.1186 - val_acc: 0.8174
Epoch 3/30
702/703 [============================>.] - ETA: 0s - loss: 0.9168 - acc: 0.8349  
Epoch 00003: val_loss improved from 1.11857 to 0.95122, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 198s 282ms/step - loss: 0.9167 - acc: 0.8349 - val_loss: 0.9512 - val_acc: 0.8401
Epoch 4/30
702/703 [============================>.] - ETA: 0s - loss: 0.6970 - acc: 0.8601  
Epoch 00004: val_loss improved from 0.95122 to 0.86415, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 192s 273ms/step - loss: 0.6967 - acc: 0.8601 - val_loss: 0.8642 - val_acc: 0.8527
Epoch 5/30
702/703 [============================>.] - ETA: 0s - loss: 0.5359 - acc: 0.8811  
Epoch 00005: val_loss improved from 0.86415 to 0.81453, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 200s 284ms/step - loss: 0.5358 - acc: 0.8811 - val_loss: 0.8145 - val_acc: 0.8618
Epoch 6/30
702/703 [============================>.] - ETA: 0s - loss: 0.4124 - acc: 0.8982  
Epoch 00006: val_loss improved from 0.81453 to 0.80447, saving model to attention_model_new_arch_maxnorm.h5
703/703 [==============================] - 195s 278ms/step - loss: 0.4122 - acc: 0.8982 - val_loss: 0.8045 - val_acc: 0.8693
Epoch 7/30
702/703 [============================>.] - ETA: 0s - loss: 0.3316 - acc: 0.9128  
Epoch 00007: val_loss did not improve from 0.80447
703/703 [==============================] - 195s 278ms/step - loss: 0.3315 - acc: 0.9128 - val_loss: 0.8135 - val_acc: 0.8704
Epoch 8/30
702/703 [============================>.] - ETA: 0s - loss: 0.2805 - acc: 0.9236  
Epoch 00008: val_loss did not improve from 0.80447
703/703 [==============================] - 194s 276ms/step - loss: 0.2804 - acc: 0.9236 - val_loss: 0.8202 - val_acc: 0.8711
Epoch 00008: early stopping
[INFO] Early stopping at epoch: 7
```

[training_loss_new_arch_maxnorm]: artifacts/training_loss_attention_model_new_arch_maxnorm_kernel_recurrent.png

![Training loss with max_norm applied to input and recurrent weights on LSTM encoder][training_loss_new_arch_maxnorm]

Adding `max_norm` to both `kernel_constraint` and `recurrent_constraint` to the encoder layer achieves a lower loss of `0.79374`.

The validation loss curve shows a gradual decline in the loss values without any inflection points.

The training log below shows that the model is trained up to 7 epochs:
```
...

Epoch 5/30
702/703 [============================>.] - ETA: 0s - loss: 0.5287 - acc: 0.8818  
Epoch 00005: val_loss improved from 0.84816 to 0.80906, saving model to attention_model_new_arch_maxnorm2.h5
703/703 [==============================] - 195s 278ms/step - loss: 0.5286 - acc: 0.8818 - val_loss: 0.8091 - val_acc: 0.8615
Epoch 6/30
702/703 [============================>.] - ETA: 0s - loss: 0.4146 - acc: 0.8986  
Epoch 00006: val_loss improved from 0.80906 to 0.79374, saving model to attention_model_new_arch_maxnorm2.h5
703/703 [==============================] - 196s 279ms/step - loss: 0.4144 - acc: 0.8986 - val_loss: 0.7937 - val_acc: 0.8683
Epoch 7/30
702/703 [============================>.] - ETA: 0s - loss: 0.3335 - acc: 0.9124  
Epoch 00007: val_loss did not improve from 0.79374
703/703 [==============================] - 195s 278ms/step - loss: 0.3334 - acc: 0.9124 - val_loss: 0.8006 - val_acc: 0.8689
Epoch 8/30
702/703 [============================>.] - ETA: 0s - loss: 0.2773 - acc: 0.9235  
Epoch 00008: val_loss did not improve from 0.79374
703/703 [==============================] - 194s 276ms/step - loss: 0.2771 - acc: 0.9235 - val_loss: 0.8167 - val_acc: 0.8705
Epoch 00008: early stopping
```
The evaluation BLEU scores indicate an improvement compared with only dropout applied as in section 5.5:
```
BLEU-1: 0.6297
BLEU-2: 0.5091
BLEU-3: 0.4467
BLEU-4: 0.3035
```

## 5.8 Addressing Variance - Dropout on LSTM layer

[training_loss_new_arch_lstm_dropout]: artifacts/training_loss_attention_model_new_arch_lstm_dropout.png

![Training loss with dropout applied to LSTM layer][training_loss_new_arch_lstm_dropout]

Dropout is applied to the LSTM encoder layer via the `dropout` and `recurrent_dropout` option. This applies dropout to the incoming and recurrent inputs.

The validation loss is slightly higher at `0.80123`.

The BLEU scores are slightly lower compared to section 5.7:
```
BLEU-1: 0.6289
BLEU-2: 0.5087
BLEU-3: 0.4467
BLEU-4: 0.3025
```

The training log indicates that the model can be trained longer, stopping at epoch 8, with a higher validation accuracy of `0.8710` but a higher validation loss of `0.8012`

```
702/703 [============================>.] - ETA: 0s - loss: 2.0271 - acc: 0.7138         
Epoch 00001: val_loss improved from inf to 1.48493, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 191s 271ms/step - loss: 2.0262 - acc: 0.7139 - val_loss: 1.4849 - val_acc: 0.7806
Epoch 2/30
702/703 [============================>.] - ETA: 0s - loss: 1.3136 - acc: 0.7950  
Epoch 00002: val_loss improved from 1.48493 to 1.17590, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 184s 262ms/step - loss: 1.3134 - acc: 0.7951 - val_loss: 1.1759 - val_acc: 0.8120
Epoch 3/30
702/703 [============================>.] - ETA: 0s - loss: 1.0189 - acc: 0.8240  
Epoch 00003: val_loss improved from 1.17590 to 0.99810, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 184s 262ms/step - loss: 1.0188 - acc: 0.8240 - val_loss: 0.9981 - val_acc: 0.8332
Epoch 4/30
702/703 [============================>.] - ETA: 0s - loss: 0.7901 - acc: 0.8485  
Epoch 00004: val_loss improved from 0.99810 to 0.88836, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 185s 263ms/step - loss: 0.7898 - acc: 0.8486 - val_loss: 0.8884 - val_acc: 0.8500
Epoch 5/30
702/703 [============================>.] - ETA: 0s - loss: 0.6235 - acc: 0.8686  
Epoch 00005: val_loss improved from 0.88836 to 0.82984, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 184s 261ms/step - loss: 0.6234 - acc: 0.8686 - val_loss: 0.8298 - val_acc: 0.8605
Epoch 6/30
702/703 [============================>.] - ETA: 0s - loss: 0.4995 - acc: 0.8850  
Epoch 00006: val_loss improved from 0.82984 to 0.80134, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 184s 261ms/step - loss: 0.4994 - acc: 0.8850 - val_loss: 0.8013 - val_acc: 0.8680
Epoch 7/30
702/703 [============================>.] - ETA: 0s - loss: 0.4073 - acc: 0.8997  
Epoch 00007: val_loss improved from 0.80134 to 0.80123, saving model to attention_model_new_arch_dropout.h5
703/703 [==============================] - 185s 263ms/step - loss: 0.4072 - acc: 0.8998 - val_loss: 0.8012 - val_acc: 0.8710
Epoch 8/30
702/703 [============================>.] - ETA: 0s - loss: 0.3425 - acc: 0.9106  
Epoch 00008: val_loss did not improve from 0.80123
703/703 [==============================] - 184s 262ms/step - loss: 0.3423 - acc: 0.9106 - val_loss: 0.8020 - val_acc: 0.8738
Epoch 9/30
702/703 [============================>.] - ETA: 0s - loss: 0.2913 - acc: 0.9210  
Epoch 00009: val_loss did not improve from 0.80123
703/703 [==============================] - 183s 260ms/step - loss: 0.2912 - acc: 0.9211 - val_loss: 0.8020 - val_acc: 0.8754
Epoch 00009: early stopping
[INFO] Early stopping at epoch: 8
```

From the results above, it is decided to keep the model's configuration and not apply dropout on the LSTM encoder layer since it doesn't appear to make much difference to model's performance.


## 5.9 Add more training data

[training_acc_more_data]: artifacts/training_acc_attention_model_more_data.png
[training_loss_more_data]: artifacts/training_loss_attention_model_more_data.png

The training set size is increased to 50,000 with 45,000 for training; 2500 for dev; 2500 for test set.

The training process is ran for 30 epochs with batch size of 64.

Below is an excerpt of the training log:
```
702/703 [============================>.] - ETA: 1s - loss: 1.4966 - acc: 0.7946         
Epoch 00001: val_loss improved from inf to 1.07001, saving model to attention_model_more_data.h5
703/703 [==============================] - 935s 1s/step - loss: 1.4959 - acc: 0.7947 - val_loss: 1.0700 - val_acc: 0.8400
Epoch 2/30
702/703 [============================>.] - ETA: 1s - loss: 0.9206 - acc: 0.8548   
Epoch 00002: val_loss improved from 1.07001 to 0.76976, saving model to attention_model_more_data.h5
703/703 [==============================] - 927s 1s/step - loss: 0.9203 - acc: 0.8549 - val_loss: 0.7698 - val_acc: 0.8740
Epoch 3/30
702/703 [============================>.] - ETA: 1s - loss: 0.6825 - acc: 0.8802   
Epoch 00003: val_loss improved from 0.76976 to 0.60291, saving model to attention_model_more_data.h5
703/703 [==============================] - 918s 1s/step - loss: 0.6825 - acc: 0.8802 - val_loss: 0.6029 - val_acc: 0.8959
Epoch 4/30
702/703 [============================>.] - ETA: 1s - loss: 0.5035 - acc: 0.9027   
Epoch 00004: val_loss improved from 0.60291 to 0.51316, saving model to attention_model_more_data.h5
703/703 [==============================] - 913s 1s/step - loss: 0.5035 - acc: 0.9027 - val_loss: 0.5132 - val_acc: 0.9082
Epoch 5/30
702/703 [============================>.] - ETA: 1s - loss: 0.3873 - acc: 0.9175   
Epoch 00005: val_loss improved from 0.51316 to 0.46587, saving model to attention_model_more_data.h5
703/703 [==============================] - 908s 1s/step - loss: 0.3872 - acc: 0.9175 - val_loss: 0.4659 - val_acc: 0.9160
Epoch 6/30
702/703 [============================>.] - ETA: 1s - loss: 0.3078 - acc: 0.9289   
Epoch 00006: val_loss improved from 0.46587 to 0.44613, saving model to attention_model_more_data.h5
703/703 [==============================] - 909s 1s/step - loss: 0.3078 - acc: 0.9289 - val_loss: 0.4461 - val_acc: 0.9189
Epoch 7/30
702/703 [============================>.] - ETA: 1s - loss: 0.2540 - acc: 0.9371   
Epoch 00007: val_loss improved from 0.44613 to 0.42919, saving model to attention_model_more_data.h5
703/703 [==============================] - 897s 1s/step - loss: 0.2539 - acc: 0.9371 - val_loss: 0.4292 - val_acc: 0.9226
Epoch 8/30
702/703 [============================>.] - ETA: 1s - loss: 0.2128 - acc: 0.9443   
Epoch 00008: val_loss did not improve from 0.42919
703/703 [==============================] - 880s 1s/step - loss: 0.2127 - acc: 0.9443 - val_loss: 0.4314 - val_acc: 0.9234
Epoch 9/30
702/703 [============================>.] - ETA: 1s - loss: 0.1832 - acc: 0.9502   
Epoch 00009: val_loss improved from 0.42919 to 0.42088, saving model to attention_model_more_data.h5
703/703 [==============================] - 882s 1s/step - loss: 0.1831 - acc: 0.9502 - val_loss: 0.4209 - val_acc: 0.9266
Epoch 10/30
702/703 [============================>.] - ETA: 1s - loss: 0.1587 - acc: 0.9553   
Epoch 00010: val_loss did not improve from 0.42088
703/703 [==============================] - 883s 1s/step - loss: 0.1586 - acc: 0.9553 - val_loss: 0.4273 - val_acc: 0.9269
Epoch 11/30
702/703 [============================>.] - ETA: 1s - loss: 0.1445 - acc: 0.9585   
Epoch 00011: val_loss did not improve from 0.42088
703/703 [==============================] - 914s 1s/step - loss: 0.1445 - acc: 0.9586 - val_loss: 0.4282 - val_acc: 0.9273
Epoch 00011: early stopping
[INFO] Early stopping at epoch: 10
```

![Training loss with increased training set][training_loss_more_data]

The validation loss has decreased to `0.42` with early stopping kicking in at epoch 10. The validation loss curve indicate that it has a gradual decline without any inflection points. However, the loss plateaus after epoch 3 without any sign of further decrease. 

![Training accuracy with increased training set][training_acc_more_data]

The validation accuracy shows an increase to `0.92`.

Overall the BLEU scores on the dev set also increased significantly, compared to 5.8:
```
BLEU-1: 0.6778
BLEU-2: 0.5673
BLEU-3: 0.5162
BLEU-4: 0.3978
```

## 6. Final Model

( Describes the choice of a final model, including configuration and performance. It is a good idea to demonstrate saving and loading the model and demonstrate the ability to make predictions on a holdout dataset. )

The table below summarizes the model's configuration in creating the final model:

| Hyperparameter   		| Value           	|
|--------------------	|-------------------|
| Encoder depth 			| 1 								|
| Encoder type 				| Bi-directional    |
| Decoder type 				| Uni-directional 	|
| Decoder depth 			| 1 								|
| Word embedding size | 512 							|
| Encoder size 				| 512 							|
| Decoder size 				| 1024 							|
| Attention type 			| Bahdanau 					|
| Dropout 						| 0.5 							|
| Weight constraint   | 3.0 							|

[final_model_arch]: artifacts/final_model.png

![Final model architecture][final_model_arch]
The model's architecture is shown in the diagram above.

Dropout is applied to after the word embedding layer and before the final softmax layer to reduce overfitting.

Weight constraints in the form of max norm are set on the encoder LSTM for both incoming and recurrent connections to a maximum value of 3.0. Previous experiments on using layer dropout did not contribute to any improvements.

The model is trained with all the training data at 50,000 without any splits for 6 epochs with mini-batch size of 64. 6 epochs were chosen based on the previous training log, which shows that the model starts to overfit beyond epoch 6.

The training log for the final model shows that the model has a final loss value of `0.26` and an accuracy of `0.93`:
```
781/781 [==============================] - 1083s 1s/step - loss: 1.4017 - acc: 0.8048   
Epoch 2/6
781/781 [==============================] - 1071s 1s/step - loss: 0.8210 - acc: 0.8644
Epoch 3/6
781/781 [==============================] - 1057s 1s/step - loss: 0.5762 - acc: 0.8923
Epoch 4/6
781/781 [==============================] - 1047s 1s/step - loss: 0.4244 - acc: 0.9117
Epoch 5/6
781/781 [==============================] - 1044s 1s/step - loss: 0.3293 - acc: 0.9246
Epoch 6/6
781/781 [==============================] - 1047s 1s/step - loss: 0.2617 - acc: 0.9347
```

The model's weights are saved in a separate file using `save_weights` API in Keras. The model structure is serialized to JSON file format using `model.to_json()`. The model is de-serialized using `model_from_json` after reading in the architecture file and the weights loaded using `model.load_weights`. Both are demonstrated in `predict.py`

The final model's accuracy is evaluated against an unseen test set, with the following BLEU scores:
```
BLEU-1: 0.6630
BLEU-2: 0.5516
BLEU-3: 0.4999
BLEU-4: 0.3797
```

The BLEU scores are lower to the development model in section 5.9. 

The `predict.py` script can produce translations from a given input sentence e.g
```
python predict.py -m final_model.h5 -s 'My name is Tom.'
```


## 7. Extensions

( Describes areas that were considered but not addressed in the project that could be explored in the future. )

* The models are tied to specific input lengths based on the dataset. Try to include longer sentences in the training dataset.

* Try a different network architecture i.e. GMNT; different attention mechanism

* Increase the number of layers for the encoder and decoder to create more representational capacity.

* Try reducing the validation loss by using more regularization techniques such as gradient clipping; different learning rate.

* Remove duplicate sentences from dataset

* Truncate the vocab by removing words with low occurences and replacing them with `UNK`


## 8. Resources
* [Deep Learning with NLP](https://machinelearningmastery.com/deep-learning-for-nlp/)

* [Sequence Model course from deep learning specialization](https://www.coursera.org/learn/nlp-sequence-models)
