# import all libraries
# tensorflow.keras for tf 2.0
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from data_prep import get_data_split
from matplotlib import pyplot as plt
import matplotlib.style as style
style.use('seaborn')
# --------------------------------------------------------------------------------
x_train, x_val, y_train, y_val = get_data_split()

# Set training parameters 
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 1

n_samples = x_train.shape[0]
n_feats = x_train.shape[1]
n_classes = 9 

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train.values, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create Model
class ConvNet(Model):
    # layers
    def __init__(self):
        super(ConvNet, self).__init__()
        # conv layer with 32 filters, kernel size 2, Maxpooled
        self.conv1 = layers.Conv2D(conv1_filters, kernel_size=1, activation = tf.nn.relu6, padding = 'same')
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv2 = layers.Conv2D(conv2_filters, kernel_size=1, activation = tf.nn.relu6, padding = 'same')
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # flatten to fit fully connected layer
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(fc1_units)
        self.dropout = layers.Dropout(rate=0.25)

        # output layer
        self.out = layers.Dense(n_classes)

    # call the above model and set the forward pass
    def call(self, x):
        x = tf.reshape(x, [-1, n_feats, 1, 1])
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)

        return x

# Build neural network model.
conv_net = ConvNet()

def compute_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass and compute loss
        pred = conv_net(x)
        loss = compute_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update weights and bias
    optimizer.apply_gradients(zip(gradients, trainable_variables))


train_loss = []
train_acc = []

# Run training
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    run_optimization(batch_x, batch_y)
        
    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = compute_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        
        train_loss.append(loss)
        train_acc.append(acc)

        print("step: %i, loss: %f, accu: %f" % (step, loss, acc))

# Test model on validation set.
pred = conv_net(x_val.values)
val_acc = accuracy(pred, y_val)
print("Validation Accuracy: %f" % val_acc)

# plot training
fig = plt.figure(figsize=(6,5))
plt.plot(train_loss, 'r', label='Loss on training')
plt.plot(train_acc, 'b', label='Accuracy on training')
plt.text(150, 0.2,  ('Val acc:', val_acc.numpy())) 
plt.title('Training Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Iter')
plt.ylim(0)
plt.legend()
plt.show()
fig.savefig('evaluation/cnn_eval.png')


