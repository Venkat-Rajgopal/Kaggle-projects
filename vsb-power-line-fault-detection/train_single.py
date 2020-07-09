from attention import Attention
from utilities import *
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

X = np.load('data/X.npy')
y = np.load('data/y.npy')

# change data dir based on machine
local = True
if local is True:
    data_dir = 'data/'
else:
    data_dir = '../input/vsb-power-line-fault-detection/'

def lstm_model(input_shape):

    # input layer then attach attention
    inp = Input(shape=(input_shape[1], input_shape[2],))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = Attention(input_shape[1])(x)

    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation]) # or accuracy

    return model

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

model = lstm_model(X_train.shape)
model.save('attention_wo_cval.h5')

ckpt = ModelCheckpoint('weights_wo_cval.h5', save_weights_only=True, verbose=1,monitor='val_matthews_correlation', mode='max') # or val_accuracy
model.fit(X_train, y_train,batch_size=100, epochs=100, validation_data=[X_valid, y_valid], callbacks=[ckpt])

print(model.metrics_names)
model.summary()

X_test_input = np.load('data/x_test.npy')

print('Loading weights')
model.load_weights('weights_wo_cval.h5')

# make predictions
print('making predictions')
pred = model.predict(X_test_input, batch_size=100)
pred_test = []
for pred_scalar in pred:
    for i in range(3):
        pred_test.append(int(pred_scalar > 0.4))

np.save('data/pred_test.npy', pred_test)
