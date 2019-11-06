# import all libraries
# tensorflow.keras for tf 2.0
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import matplotlib.style as style
style.use('seaborn')
# --------------------------------------------------------------------------------
# Read dataframes
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# drop id column
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# reform targets as one hot encoded vectors. 
y = train['target']
y = y.values.reshape(-1,1)
cat = OneHotEncoder()
y = cat.fit_transform(y).toarray()

# drop target vector 
train = train.drop(['target'], axis=1)

# Split training samples for train and val. 
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size = 0.2)
print('Train size:', X_train.shape)
print('Val size:', X_val.shape)
print('Train labels:', y_train.shape)
print('Val labels:', y_val.shape)

n_samples = X_train.shape[0]
n_feats = X_train.shape[1]
n_classes = y_train.shape[1]
# --------------------------------------------------------------------------------
# Dense Network
model = Sequential()
model.add(Reshape((n_feats, 1), input_shape=(n_feats,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(n_classes, activation='softmax'))
print(model.summary())

# save the best model during training
callbacks_list = [ModelCheckpoint(
        filepath = 'best_mods/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor = 'val_loss', mode = 'min', save_best_only=True),
        EarlyStopping(monitor='accuracy', patience=25)
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=100, callbacks = callbacks_list, verbose=1, validation_data=(X_val, y_val))

fig = plt.figure(figsize=(6,5))
plt.plot(history.history['accuracy'], 'r', label='Accuracy on training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy on validation data')
plt.plot(history.history['loss'], 'r--', label='Loss on training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss on validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()
fig.savefig('evaluation/fc_eval.png')