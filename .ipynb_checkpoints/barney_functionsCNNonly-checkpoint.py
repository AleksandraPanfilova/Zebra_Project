#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:46:48 2022

@author: barnabyemmens
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import re
import os
# from tqdm import tqdm 
# import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savefig = ''):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=16)
        plt.yticks(tick_marks, target_names, fontsize=16)
        

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=13,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel(f'Predicted label\n\n accuracy={accuracy:0.3f}; misclass={misclass:0.3f}', fontsize=16)
    cbar.ax.set_ylabel('Number of items',  labelpad=20, rotation=270, fontsize=16)   
    
    
    if savefig: plt.savefig(savefig, bbox_inches='tight')
    
    plt.show()

def check_class_complete(kfold,X,y):
    fold_no = 1
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if len(np.unique(y_train)) != len(np.unique(y)):
            print("FOLD "+str(fold_no)+": Class missing from fold train data:")
        elif len(np.unique(y_test)) != len(np.unique(y)):
            print("FOLD "+str(fold_no)+": Class missing from fold validation data.")
        fold_no += 1
    print("ALL CLASSES PRESENT")
    
def check_class_complete_gen(y_train, y_test,y):
    if len(np.unique(y_train)) != len(np.unique(y)):
        print("Class missing from train data")
    elif len(np.unique(y_test)) != len(np.unique(y)):
        print("Class missing from validation data.")
    else:
        print("OK")
    
def standardizeimg(img, mu, sigma):
    return (img-mu)/(sigma).astype(np.float32)

def format_for_CNN(X_train,X_test,y_train,y_test):
    # save for scaling test data
    mu_train = np.mean(X_train)
    sigma_train = np.std(X_train)

    # Standardize pixel distribution to have zero mean and unit variance
    train_images = standardizeimg(img=X_train, mu=mu_train, sigma=sigma_train)
    val_images = standardizeimg(img=X_test, mu=np.mean(X_test), sigma=np.std(X_test))

    # adapt to format required by tensorflow; Using channels_last --> (n_samples, img_rows, img_cols, n_channels)
    img_rows, img_cols = X_train.shape[1], X_train.shape[2] # input image dimensions
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # convert class vectors to binary class matrices - one hot encoding
    onehot_t = pd.get_dummies(y_train)
    label_list = onehot_t.columns
    y_train = onehot_t.to_numpy()

    onehot_v = pd.get_dummies(y_test)
    y_test = onehot_v.to_numpy()
    return X_train,X_test,y_train,y_test,img_rows,img_cols