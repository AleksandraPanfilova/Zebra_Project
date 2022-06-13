from barney_functions import *
import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import re
import os
from tqdm import tqdm 
import soundfile as sf
from sklearn.model_selection import train_test_split

def calc_melstft_inaug(augmented_samples):
    n_samples = augmented_samples.shape[0]
    test_shape = np.abs(librosa.feature.melspectrogram(y=augmented_samples[0,:])).shape
    sftfs_0 = test_shape[0]
    sftfs_1 = test_shape[1]
    audio_mel = np.zeros((n_samples,sftfs_0,sftfs_1))
    for j in tqdm(range(n_samples)):
        audio_mel[j,:,:] = np.abs(librosa.feature.melspectrogram(y=augmented_samples[j,:]))
    return audio_mel


audio_path = 'zebra audio sample_Bing_413/all/'
audio_path_long = 'Audio Files All/'
spreadsheet_path = 'Zebras.Assumption.data_Bing_413 .xlsx'
df = add_labels_to_excel(spreadsheet_path)
zebra_labels = df['label']
audio_size,audio_list_short,labels_short,_ = length_finder(audio_path,df,97)
zebra_labels = pd.Series(data=labels_short, index=[np.arange(len(labels_short))])
audio_files = pad(audio_list_short,audio_size)

train_images, TEST_images, train_y, TEST_y = train_test_split(audio_files, zebra_labels, test_size=0.25, random_state=42)

print(TEST_images.shape)

balanced, balanced_labels = balancer(train_images,train_y)
augmented_audio = augment_audio_faster_smaller(balanced)
augmented_TEST = augment_audio_faster_smaller(TEST_images)
print(augmented_audio.shape)

print("Augmented dataset size: ",augmented_audio.nbytes/(1e9), 'GB')

mels = calc_melstft(augmented_audio)
TEST_mels = calc_melstft_inaug(augmented_TEST[0])
mels_norm_db = librosa.util.normalize(librosa.power_to_db(mels))
TEST_mels_norm_db = librosa.util.normalize(librosa.power_to_db(TEST_mels))
print(mels_norm_db.shape)
print(TEST_mels_norm_db.shape)
print("Augmented dataset size: ",mels_norm_db.nbytes/(1e9), 'GB')

mels_1D = np.zeros(mels_norm_db[0].shape)
mels_labels = balanced_labels
mels_1D = mels_norm_db[0]
print("Flattening")
for i in range(1,mels_norm_db.shape[0]):
    mels_1D = np.concatenate((mels_1D,mels_norm_db[i]), axis = 0)
    mels_labels = np.concatenate((mels_labels,balanced_labels))
print("Final data shape:",mels_1D.shape)
print("Final labels shape:",mels_labels.shape)
print("Final dataset size: ",mels_norm_db.nbytes/(1e9), 'GB')

np.savez_compressed('mels_train',data=mels_1D)
np.savez_compressed('mels_train_labels',data=mels_labels)
np.savez_compressed('test-inaug',data=TEST_mels_norm_db)
np.savez_compressed('test-inaug_labels',data=TEST_y)