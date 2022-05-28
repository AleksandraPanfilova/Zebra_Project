# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:54:21 2022

@author: User
"""
# IMPORTS

import h5py
import pandas as pd
import tensorflow as tf
import lightgbm
import numpy as np
import re
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD

#%% FUNCTIONS

def add_labels_to_excel(filename, sheet_name=2):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    labels = []
    for filename in df['file']:
        label = re.search("squeal|whinnie|softsnort|snort", filename)
        if label:
            label = label.group(0)
            # if label == 'squeal':
            #     num_label = 0
            # elif label == 'whinnie':
            #     num_label = 1
            # elif label == 'softsnort':
            #     num_label = 2
            # else:
            #     num_label = 3
            # labels.append(num_label)
            labels.append(label)
    df['label'] = labels
    return df

def split_dataset(X, y, test_total_ratio, val_train_ratio):
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_total_ratio, random_state=0, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=val_train_ratio, random_state=0, shuffle=True)
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error (Loss)')
    plt.legend()
    plt.grid(True)

#%%

df = add_labels_to_excel('Zebras_Assumption_data.xlsx')

X = df[df.columns.drop('file').drop('label')]
y = df['label']

# Fractions must add up to 1.0
train_frac = 0.70
val_frac = 0.01
test_frac = 0.29
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, test_frac, val_frac/(train_frac + val_frac))

#%% FEATURE IMPORTANCE
lgbm_model = lightgbm.LGBMClassifier()
lgbm_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=20, eval_metric='logloss')
print(lgbm_model.feature_importances_)

#%% SCALING
X_scaler = StandardScaler()

# Fit on Training Data
X_scaler.fit(X_train.values)

# Transform Training, Validation and Testing data
X_train = X_scaler.transform(X_train.values)
X_val = X_scaler.transform(X_val.values)
X_test = X_scaler.transform(X_test.values)

lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_val = lb.fit_transform(y_val)
y_test = lb.fit_transform(y_test)

#%% define the keras model
# callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=0,
#     patience=5,
#     verbose=0,
#     mode='auto',
#     baseline=None,
#     restore_best_weights=False
# )

model =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(9,)),
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(8,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#%%
history = model.fit(X_train, y_train, epochs=1000, batch_size=32)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

#%%
plot_loss(history)

