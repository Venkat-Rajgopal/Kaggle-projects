"""
Train and predict on test set
"""

from sklearn.model_selection import StratifiedKFold
from attention import Attention
from utilities import *
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import pandas as pd

n_splits = 5
X = np.load('data/X.npy')
y = np.load('data/y.npy')
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019).split(X, y))

def lstm_model(input_shape):

    # input layer then attach attention
    inp = Input(shape=(input_shape[1], input_shape[2],))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Attention(input_shape[1])(x)

    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # matthews_correlation

    return model

preds_val = []
y_val = []

# cross validation loop
for idx, (train_idx, val_idx) in enumerate(splits):

    print('Training for Split:', idx)

    # create partitions
    train_x, train_y, val_x, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    model = lstm_model(train_x.shape)
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_accuracy', mode='max') # or val_accuracy
    model.fit(train_x, train_y, batch_size=128, epochs=50, validation_data=[val_x, val_y], callbacks=[ckpt])

    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_x, batch_size=128))

    model.save('attention_model.h5')

    # and the val true y (for metric cal)
    y_val.append(val_y)


preds_val = np.concatenate(preds_val)[...,0]
y_val = np.concatenate(y_val)

print('Unique Validation Predictions:', np.unique(preds_val))
print('y val values:', np.unique(y_val))

print('True y shape', y_val.shape)
print('Predicted y shape', preds_val.shape)

best_thres = threshold_search(y_val, preds_val)['threshold']
best_score = threshold_search(y_val, preds_val)['matthews_correlation_score']

print('Best Threshold', best_thres, 'Best score', best_score)

# make  predictions
x_test_input = np.load('data/x_test.npy')

preds_test = []
for i in range(n_splits):
    model = load_model('attention_model.h5')
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict(x_test_input, batch_size=128, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)

preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_thres).astype(np.int)

preds_test.to_csv('final_predictions', index=False)
print(preds_test.shape)
print("Predictions counts ", preds_test.unique())
