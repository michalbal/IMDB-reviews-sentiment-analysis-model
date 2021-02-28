
##########################
# Code for Ex. #2 in IDL #
##########################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
from tensorflow.python.keras import backend as K
import tensorflow as tf
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import sys
sys.setrecursionlimit(2500)

import os

import loader as ld

train_texts, train_labels, test_texts, test_labels, test_ascii, embedding_matrix, MAX_LENGTH, MAX_FEATURES = ld.get_dataset()

#####################
# Execusion options #
#####################

TRAIN = False

RECR = False # recurrent netowrk (RNN/GRU) or a non-recurrent network

ATTN = True # use attention layer in global sum pooling or not
LSTM = False # use LSTM or otherwise RNN

BATCH_SIZE = 128


# Getting activations from model

def get_act(net, input, name):
    sub_score = [layer for layer in model.layers if name in layer.name][0].output
    # functor = K.function([test_texts]+ [K.learning_phase()], sub_score)

    OutFunc = K.function([net.input], [sub_score])
    return OutFunc([test_texts])[0]


# RNN Cell Code

def RNN(dim ,x):

    # Learnable weights in the cell
    Wh = layers.Dense(dim, use_bias=False)
    Wx = layers.Dense(dim)

    # unstacking the time axis
    x = tf.unstack(x ,axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))
    relu = layers.ReLU()
    for i in range(len(x)):
        a = Wx(x[i])
        if i != 0:
            h_i = H[i - 1]
        else:
            h_i = h

        b = Wh(h_i)
        mid_val = a + b
        h = relu(mid_val)

        H.append(h)

    H = tf.stack(H ,axis=1)

    return h, H

# GRU Cell Code

def GRU(dim ,x):

    # Learnable weights in the cell
    Wzx = layers.Dense(dim)
    Wzh = layers.Dense(dim, use_bias=False)

    Wrx = layers.Dense(dim)
    Wrh = layers.Dense(dim, use_bias=False)

    Wx = layers.Dense(dim)
    Wh = layers.Dense(dim, use_bias=False)

    # unstacking the time axis
    x = tf.unstack(x ,axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))

    for i in range(len(x)):
        # -- missing code --
        if i != 0:
            h_i = H[i - 1]
        else:
            h_i = h
        z = tf.sigmoid(Wzx(x[i])+Wzh(h_i))
        r = tf.sigmoid(Wrx(x[i])+Wrh(h_i))
        ht = tf.tanh(Wx(x[i])+Wh(r*h_i))
        h =(1-z)*h_i+z*ht

        H.append(h)

    H = tf.stack(H ,axis=1)

    return h, H

# (Spatially-)Restricted Attention Layer
# k - specifies the -k,+k neighbouring words

def restricted_attention(x ,k):
    dim = x.shape[2]

    Wq = layers.Dense(dim)
    Wk = layers.Dense(dim)

    wk = Wk(x)

    paddings = tf.constant([[0, 0 ,], [k, k], [0 ,0]])
    pk = tf.pad(wk, paddings)
    pv = tf.pad(x, paddings)

    keys = []
    vals = []
    for i in range(-k , k +1):
        keys.append(tf.roll(pk ,i ,1))
        vals.append(tf.roll(pv ,i ,1))

    keys = tf.stack(keys ,2)
    keys = keys[: ,k:-k ,: ,:]
    vals = tf.stack(vals ,2)
    vals = vals[: ,k:-k ,: ,:]

    # -- missing code --

    query = Wq(x)
    reshaped_keys = tf.reshape(keys, (-1, 100, 2*k+1, 100))
    reshaped_query = tf.reshape(query, (-1, 100, 100, 1))
    dot_product = tf.matmul(reshaped_keys,
                            reshaped_query) / (dim ** 0.5)

    atten_weights = layers.Dense(1, activation='softmax', name='atten_weights')(dot_product)#MAKE SURE you have ,name= "atten_weights" in the attention weight step
    val_out = tf.multiply(vals, atten_weights)
    val_out = tf.reduce_sum(val_out, axis=2)

    return x + val_out


# Building Entire Model
def build_model():
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedding_layer = layers.Embedding(MAX_FEATURES, 100,
                                       weights=[embedding_matrix],
                                       input_length=MAX_LENGTH,
                                       trainable=False)

    # embedding the words into 100 dim vectors

    x = embedding_layer(sequences)

    if not RECR:

        # non recurrent networks

        if ATTN:
            # attention layer
            x = restricted_attention(x, k=5)

        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(5, activation='relu')(x)

        # Question 2
        # x = layers.Dense(1, name="sub_score")(x)
        # global_average_pooling = tf.reduce_mean(x, [1, 2])
        # x = tf.sigmoid(global_average_pooling)

        # Question 3
        x = layers.Dense(2, name="sub_score")(x)

        # multiply by weights

        weights = layers.Dense(1, activation='softmax')(x[:, :, 1])

        x = tf.multiply(x[:, :, 0], weights)

        global_average_pooling = tf.reduce_mean(x, [1])
        x = tf.sigmoid(global_average_pooling)

        predictions = x

    else:
        # recurrent networks
        if LSTM:
            x, _ = GRU(128, x)
        else:
            x, _ = RNN(128, x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        predictions = x

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def plot_loss(epochs, train_loss, test_loss):
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.title('Word sum - Train and Test loss as function of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


model = build_model()

checkpoint_path = "model_save/cp.ckpt"

if TRAIN:
    print("Training")

    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    # print(model.summary())

    model.fit(
        train_texts,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=15,
        validation_data=(test_texts, test_labels), callbacks=[cp_callback])
    plot_loss(model.history.epoch, model.history.history['loss'],model.history.history['val_loss'])
else:
    model.load_weights(checkpoint_path)

#############
# test code #
#############

print("Example Predictions:")
preds = model.predict(test_texts)

if not RECR:
    sub_score = get_act(model, test_texts, "sub_score")

for i in range(4):

    print("-" * 20)

    if not RECR:
        # print words along with their sub_score

        num = min((len(test_ascii[i]), 100))
        print(test_ascii[i])
        print("sentence prediction ", preds[i])
        for k in range(num):
            print("Word number ", k, " is ", test_ascii[i][k], " sub_score: ", sub_score[i][k][0])


        print("\n")
    else:
        print(test_ascii[i])
        print(preds[i])

    if preds[i] > 0.5:
        print("Positive")
    else:
        print("Negative")
    print("-" * 20)

print('Accuracy score: {:0.4}'.format(
    accuracy_score(test_labels, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))