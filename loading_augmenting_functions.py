#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:46:48 2022

@author: barnabyemmens
"""


import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import re

def add_labels_to_excel(data_path):
    """Returns dataframe of spreadsheet with labels"""
    df = pd.read_excel(data_path, sheet_name=2)
    labels = []
    for filename in df['file']:
        label = re.search("squeal|whinnie|softsnort|snort", filename)
        if label:
            label = label.group(0)
            labels.append(label)
    df['label'] = labels
    return df

def gen_audio_array(folder_path,df):
    """ Loads and pads audio files to be of uniform shape"""
    audio_size = 0
    for i in range(len(df['file'])):
        audio, _ = librosa.load(folder_path+df['file'][i])
        if len(audio)>audio_size:
            audio_size = len(audio)
            index_longest = i
    audio_files = np.zeros((413,audio_size))

    for i in range(len(df['file'])):
        audio, _ = librosa.load(folder_path+df['file'][i])
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) == 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files[i,:] = audio_padded
    return audio_files, audio_size, index_longest
    
def buffer(audio,max_frac_shift):
    shift_max = int(audio.shape[0]*max_frac_shift)
    buffer_array = np.zeros(shift_max)
    audio_buff = np.append(audio,buffer_array, axis=0)
    audio_buff = np.append(buffer_array,audio_buff, axis=0)
    return audio_buff

def shifter(audio,max_frac_shift):
    shift_max = int(audio.shape[0]*max_frac_shift)
    audio_shift = np.roll(audio, random.randint(-shift_max,shift_max))
    return audio_shift

def louder(audio, max_frac_louder):
    max_change = int(max_frac_louder*100)
    random_change = random.randint(100-max_change,max_change)/100
    return audio*random_change

def plot_mel(audio):
    file1_mel = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=audio))
    plt.imshow(file1_mel,aspect='auto')
    plt.colorbar(label='dB')
    #plt.xlabel('Frequency Bin')
    #plt.ylabel('Frame')
    plt.show()
    
def augment_audio(audio_files):
    time_shift = 0.1 # fractional ranges
    vol_shift = 1.1 
    n_samples = audio_files.shape[0]
    width_buffer = int(audio_files.shape[1]*(1+time_shift*2))
    buffer_shape = (n_samples,width_buffer)
    

    audio_files_norm = np.zeros(audio_files.shape) # assume beyond this point (unlabelled)
    audio_files_buff = np.zeros(buffer_shape)
    audio_files_shift1 = np.zeros(buffer_shape)
    audio_files_shift2 = np.zeros(buffer_shape)
    audio_files_loud = np.zeros(buffer_shape) # random intensity shift (whole file)
    audio_files_noisy = np.zeros(audio_files.shape)
    audio_files_noisy_buff = np.zeros(buffer_shape)

    for i in range(n_samples):
        audio_files_norm[i,:] = librosa.util.normalize(audio_files[i,:]) 
        audio_files_buff[i,:]  = buffer(audio_files_norm[i,:],time_shift)
        audio_files_shift1[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_shift2[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_loud[i,:] = louder(audio_files_buff[i,:],vol_shift)
        
    noise = np.random.normal(0,0.01,39923)

    ## For noise in buffer:

    for i in range(n_samples):
        for j in range(39923):
            if audio_files_norm[i,j] != 0:
                audio_files_noisy[i,j] =  audio_files_norm[i,j] + noise[j]
            else:
                audio_files_noisy[i,j] =  audio_files_norm[i,j]
        audio_files_noisy_buff[i,:] = buffer(audio_files_noisy[i,:],time_shift)   
        
    data_for_NN = np.zeros((5,n_samples,47907))
    data_for_NN[0,:,:] = audio_files_buff # All normalised 1st
    data_for_NN[1,:,:] = audio_files_shift1
    data_for_NN[2,:,:] = audio_files_shift2
    data_for_NN[3,:,:] = audio_files_noisy_buff
    data_for_NN[4,:,:] = audio_files_loud
        
    return data_for_NN

def plot_sample(augmented_samples,sample_index):
    plt.plot(augmented_samples[0,sample_index,:],label='buff')
    plt.plot(augmented_samples[1,sample_index,:],label='shift1')
    plt.plot(augmented_samples[2,sample_index,:],label='shift2')
    plt.plot(augmented_samples[3,sample_index,:],label='noisy')
    plt.plot(augmented_samples[4,sample_index,:],label='loud')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
def hsr_loader(folder_path,df,audio_size):
    hsr=22050*2
    audio_size_hsr = int(audio_size*(hsr/22050))
    audio_files_hsr = np.zeros((413,audio_size_hsr))
    for i in range(len(df['file'])):
        audio, _ = librosa.load((folder_path+df['file'][i]), sr=hsr)
        padding_amount = int((audio_size_hsr - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) != 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files_hsr[i,:] = audio_padded
    return audio_files_hsr

def lsr_loader(folder_path,df,audio_size):
    lsr=22050/2
    audio_size_lsr = int(audio_size*(lsr/22050))+1
    audio_files_lsr = np.zeros((413,audio_size_lsr))
    for i in range(len(df['file'])):
        audio, _ = librosa.load((folder_path+df['file'][i]), sr=lsr)
        padding_amount = int((audio_size_lsr - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) != 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files_lsr[i,:] = audio_padded
    return audio_files_lsr

def calc_stft(augmented_samples):
    n_augments = augmented_samples.shape[0]
    n_samples = augmented_samples.shape[1]
    test_shape = np.abs(librosa.stft(y=augmented_samples[0,0,:])).shape
    sftfs_0 = test_shape[0]
    sftfs_1 = test_shape[1]
    audio_stfts = np.zeros((n_augments,n_samples,sftfs_0,sftfs_1))
    for i in range(n_augments):
        for j in range(n_samples):
            audio_stfts[i,j,:,:] = np.abs(librosa.stft(y=augmented_samples[i,j,:]))
    return audio_stfts

def calc_melstft(augmented_samples):
    n_augments = augmented_samples.shape[0]
    n_samples = augmented_samples.shape[1]
    test_shape = np.abs(librosa.feature.melspectrogram(y=augmented_samples[0,0,:])).shape
    sftfs_0 = test_shape[0]
    sftfs_1 = test_shape[1]
    audio_mel = np.zeros((n_augments,n_samples,sftfs_0,sftfs_1))
    for i in range(n_augments):
        for j in range(n_samples):
            audio_mel[i,j,:,:] = np.abs(librosa.feature.melspectrogram(y=augmented_samples[i,j,:]))
    return audio_mel

def spec_plot(stft_sample):
    plt.imshow(librosa.amplitude_to_db(stft_sample),aspect='auto', origin='lower')
    plt.xlabel("Time bins")
    plt.ylabel('Frequency bins')
    
    
    
    
    
    
    
    