# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:30:58 2022

@author: User
"""

#%%
import h5py
import pandas as pd
import tensorflow as tf
import numpy as np
import lightgbm
import re
import umap
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

#%%
def add_labels_to_excel(filename, sheet_name=2):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    labels = []
    for file in df['file']:
        label = re.search("squeal|whinnie|softsnort|snort", file)
        if label:
            label = label.group(0)
            if label == "squeal":
                labels.append("squeal")
            elif label == "whinnie":
                labels.append("whinnie")
            elif label == 'snort':
                labels.append("snort")
            elif label == 'softsnort':
                labels.append("softsnort")
    df['label'] = labels
    return df

#%%
directory_file = r"./Zebras_Assumption_data.xlsx"
df = add_labels_to_excel(directory_file)

X = df[df.columns.drop('file').drop('label')]
# .drop('q25').drop('q50').drop('q75').drop('fpeak').drop('am.extent').drop('harmonicity').drop('am.rate')
y = df['label']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

#%% FEATURE IMPORTANCE

fi_model = lightgbm.LGBMClassifier()

if val_present == True:
    fi_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=20, eval_metric='logloss')
else:
    fi_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=20, eval_metric='logloss')
feature_importance = fi_model.feature_importances_
print(feature_importance)

total = 0
for i in feature_importance:
    total += i
average = total / len(feature_importance)

feature_importance_series = pd.Series(feature_importance, index=X_train.columns)
impt_features_list = feature_importance_series[feature_importance >= average].index
X_train = X_train[impt_features_list]
X_test = X_test[impt_features_list]
if val_present:
    X_val = X_val[impt_features_list]
    
print(impt_features_list)



#%%
parameters_GridSearch = {'learning_rate': [0.001, 0.01, 0.1],
                         'max_depth': [1, 2, 3, 4, 5],
                         'n_estimators': [50, 100, 150],
                         'num_leaves': [5, 10, 20, 30, 40]
                        }

grid_clf = lightgbm.LGBMClassifier(random_state=42)

GridSearch = GridSearchCV(grid_clf, 
                          parameters_GridSearch, 
                          cv=5, 
                          return_train_score=True, 
                          refit=True, 
                          verbose=5
                         )

GridSearch.fit(X_train, y_train)
GridSearch_results = pd.DataFrame(GridSearch.cv_results_) 
print("Grid Search: \tBest parameters: ", GridSearch.best_params_, f", Best scores: {GridSearch.best_score_:.4f}\n")
clf_GridSearch = GridSearch.best_estimator_
accuracy_GridSearch = accuracy_score(y_test, clf_GridSearch.predict(X_test))
# print(f'Accuracy Manual:      {accuracy_manual:.4f}')
print(f'Accuracy Grid Search: {accuracy_GridSearch:.4f}')

#%%
clf = lightgbm.LGBMClassifier(learning_rate=0.1, max_depth=1, n_estimators=100, num_leaves=5, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%%
# cf_matrix = confusion_matrix(y_test, y_pred)
# print(cf_matrix)
# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# ax.set_title('Seaborn Confusion Matrix with labels\n\n')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['snort','squeal'])
# ax.yaxis.set_ticklabels(['snort','squeal'])

