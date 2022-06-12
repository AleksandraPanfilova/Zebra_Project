#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:46:48 2022

@author: barnabyemmens
"""

import librosa
import librosa.display
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import re
import os
from tqdm import tqdm 
import soundfile as sf
from playsound import playsound
import itertools

def gen_labels(filenames):
    labels = []
    for filename in filenames:
        label = re.search("squeal|whinnie|softsnort|snort", filename)
        if label:
            label = label.group(0)
            labels.append(label)
    return labels

def gen_audio_array_noexcel(folder_path):
    """ Loads and pads audio files to be of uniform shape"""
    filenames = os.listdir(folder_path)
    audio_size = 0
    n_samples = len(filenames)
    print('Finding longest file (Worse Labels)')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+filenames[i])
        if len(audio)>audio_size:
            audio_size = len(audio)
            longest_file = filenames[i]
            
    audio_files = np.zeros((n_samples,audio_size))
    
    print('Loading files')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+filenames[i])
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) == 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files[i,:] = audio_padded
    return audio_files, audio_size, filenames, longest_file

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
    n_samples = len(df['file'])
    print('Finding longest file (Better Labels)')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+df['file'][i])
        if len(audio)>audio_size:
            audio_size = len(audio)
            index_longest = i
            
    audio_files = np.zeros((n_samples,audio_size))

    print('Loading files')
    for i in tqdm(range(len(df['file']))):
        audio, _ = librosa.load(folder_path+df['file'][i])
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) == 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files[i,:] = audio_padded
    return audio_files, audio_size, index_longest

def msr_log(folder_path,df):
    """ Loads and pads audio files to be of uniform shape"""
    audio_size = 0
    n_samples = len(df['file'])
    print('Finding longest file (Better Labels)')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+df['file'][i])
        if len(audio)>audio_size:
            audio_size = len(audio)
            index_longest = i
            
    audio_files = np.zeros((n_samples,audio_size))

    print('Loading files')
    for i in tqdm(range(len(df['file']))):
        audio, _ = librosa.load(folder_path+df['file'][i])
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) == 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files[i,:] = audio_padded
    audio_files = librosa.amplitude_to_db(audio_files)
    return audio_files, audio_size, index_longest

def length_finder(folder_path,df,percentile):
    """ Loads and pads audio files to be of uniform shape"""
    n_samples = len(df['file'])
    labels = df['label']
    lengths = np.zeros((n_samples))
    idxs = np.arange(0,n_samples,1)
    audio_list = []
    print('Finding file lengths')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+df['file'][i],sr=22050)
        audio_list.append(audio)
        lengths[i] = len(audio)
    ninetieth_perc = np.percentile(lengths,percentile)
    audio_list_short = []
    labels_short = []
    lengths_short = []
    removed_audio = []
    removed_labels = []
    idxs_short = []
    print("Reducing to "+str(percentile)+"th percentile")
    for i in tqdm(range(n_samples)):
        if len(audio_list[i])<ninetieth_perc:
            audio_list_short.append(audio_list[i])
            labels_short.append(labels[i])
            lengths_short.append(len(audio_list[i]))
            idxs_short.append(idxs[i])
        else:
            removed_audio.append(audio_list[i])
            removed_labels.append(labels[i])      
    max_length = int(max(lengths_short))
    idx_longest = np.argmax(np.array(lengths_short))
    print("Data size reduction: ",np.around(ninetieth_perc/max_length,3))
    print("Removed classes: ",np.unique(removed_labels))
    print("New max length: ",max(lengths_short))
    print("Number of samples removed:",len(removed_labels))
    plt.scatter(idxs,lengths,label="Removed")
    plt.scatter(idxs_short,lengths_short,label=str(percentile)+"th percentile")
    plt.xlabel("Index")
    plt.ylabel("Length")
    plt.legend()
    plt.show()
    return max_length,audio_list_short,labels_short, idx_longest



# def louder(audio, max_frac_louder):
#     random_change = random.randint()
#     return audio*random_change 

def plot_mel(audio):
    file1_mel = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=audio))
    plt.imshow(file1_mel,aspect='auto')
    plt.colorbar(label='dB')
    #plt.xlabel('Frequency Bin')
    #plt.ylabel('Frame')
    plt.show()
    
def augment_audio(audio_files):
    time_shift = 1.1 # fractional ranges
    vol_shift = 1.1 
    n_samples = audio_files.shape[0]
    width_no_buffer = audio_files.shape[1]
    width_buffer = len(buffer(audio_files[0,:],time_shift))
    buffer_shape = (n_samples,width_buffer)
    
    audio_files_norm = np.zeros(audio_files.shape) # assume beyond this point (unlabelled)
    audio_files_buff = np.zeros(buffer_shape)
    audio_files_shift1 = np.zeros(buffer_shape)
    audio_files_shift2 = np.zeros(buffer_shape)
    audio_files_loud = np.zeros(buffer_shape) # random intensity shift (whole file)
    audio_files_noisy = np.zeros(audio_files.shape)
    audio_files_noisy_buff = np.zeros(buffer_shape)
    
    print("Augmenting Non-noise")
    for i in tqdm(range(n_samples)):
        audio_files_norm[i,:] = librosa.util.normalize(audio_files[i,:]) 
        audio_files_buff[i,:]  = buffer(audio_files_norm[i,:],time_shift)
        audio_files_shift1[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_shift2[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_loud[i,:] = louder(audio_files_buff[i,:],vol_shift)
        
    noise = np.random.normal(0,0.01,width_no_buffer)

    ## For noise in buffer:

    print("Making noise")
    for i in tqdm(range(n_samples)):
        for j in range(width_no_buffer):
            if audio_files_norm[i,j] != 0:
                audio_files_noisy[i,j] =  audio_files_norm[i,j] + noise[j]
            else:
                audio_files_noisy[i,j] =  audio_files_norm[i,j]
        audio_files_noisy_buff[i,:] = buffer(audio_files_noisy[i,:],time_shift)   
        
    n_augments = 5
        
    data_for_NN = np.zeros((n_augments,n_samples,width_buffer))
    data_for_NN[0,:,:] = audio_files_buff # All normalised 1st
    data_for_NN[1,:,:] = audio_files_shift1
    data_for_NN[2,:,:] = audio_files_shift2
    data_for_NN[3,:,:] = audio_files_noisy_buff
    data_for_NN[4,:,:] = audio_files_loud
        
    return data_for_NN

def augment_audio_faster(audio_files):
    time_shift = 0.1 # fractional ranges
    vol_shift = 1.1 
    n_samples = audio_files.shape[0]
    #width_no_buffer = audio_files.shape[1]
    width_buffer = len(buffer(audio_files[0,:],time_shift))
    buffer_shape = (n_samples,width_buffer)
    
    audio_files_norm = np.zeros(audio_files.shape) # assume beyond this point (unlabelled)
    audio_files_buff = np.zeros(buffer_shape)
    audio_files_shift1 = np.zeros(buffer_shape)
    audio_files_shift2 = np.zeros(buffer_shape)
    audio_files_loud = np.zeros(buffer_shape) # random intensity shift (whole file)
    audio_files_noisy = np.zeros(buffer_shape)

    
    noise = np.random.normal(0,0.01,width_buffer)
    
    print("Augmenting")
    for i in tqdm(range(n_samples)):
        audio_files_norm[i,:] = librosa.util.normalize(audio_files[i,:]) 
        audio_files_buff[i,:]  = buffer(audio_files_norm[i,:],time_shift)
        audio_files_shift1[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_shift2[i,:] = shifter(audio_files_buff[i,:],time_shift)
        audio_files_loud[i,:] = louder(audio_files_buff[i,:],vol_shift)
        audio_files_noisy[i,:] =  audio_files_buff[i,:] + noise

    n_augments = 5
        
    data_for_NN = np.zeros((n_augments,n_samples,width_buffer))
    data_for_NN[0,:,:] = audio_files_buff # All normalised 1st
    data_for_NN[1,:,:] = audio_files_shift1
    data_for_NN[2,:,:] = audio_files_shift2
    data_for_NN[3,:,:] = audio_files_noisy
    data_for_NN[4,:,:] = audio_files_loud
        
    return data_for_NN

def fast_noise(audio, noisiness, width_buffer):
    noise = np.random.normal(0,noisiness,width_buffer)
    return audio + noise

def shifter(audio,width_buffer,max_frac_shift,width_no_buffer):
    shift_max = int(audio.shape[0]*max_frac_shift)
    audio_shift = np.roll(audio, random.randint(-shift_max,shift_max))
    return audio_shift

#def buffer(audio,max_frac_shift):
#    shift_max = int(audio.shape[0]*max_frac_shift/4)
#    buffer_array = np.zeros(shift_max)
##    audio_buff = np.append(audio,buffer_array, axis=0)
#    audio_buff = np.append(buffer_array,audio_buff, axis=0)
#    return audio_buff

def buffer(audio,max_frac_shift):
    shift_max = int(audio.shape[0]*max_frac_shift)
    audio_buff = np.pad(audio,shift_max)
    return audio_buff

#def shifter(audio,max_frac_shift):
#    shift_max = int(audio.shape[0]*max_frac_shift)/2
#    audio_shift = np.roll(audio, random.randint(-shift_max,shift_max))
#    return audio_shift

def louder(audio, max_frac_louder):
    random_change = random.randint(95,105)/100
    return audio*random_change

def augment_audio_faster_smaller(audio_files):
    time_shift = 0.1 # fractional ranges
    vol_shift = 1.1 
    n_samples = audio_files.shape[0]
    width_no_buffer = audio_files.shape[1]
    width_buffer = len(buffer(audio_files[0,:],time_shift))
    buffer_shape = (n_samples,width_buffer)
    
    audio_files_norm = np.zeros(audio_files.shape) # assume beyond this point (unlabelled)
    audio_files_buff = np.zeros(buffer_shape)
    audio_files_shift1 = np.zeros(buffer_shape)
    #audio_files_shift2 = np.zeros(buffer_shape)
    audio_files_loud = np.zeros(buffer_shape) # random intensity shift (whole file)
    audio_files_noisy = np.zeros(buffer_shape)

    
    noise = np.random.normal(0,0.01,width_buffer)
    
    print("Augmenting "+str(n_samples) +" samples")
    for i in tqdm(range(n_samples)):
        audio_files_norm[i,:] = librosa.util.normalize(audio_files[i,:]) 
        #audio_files_norm[i,:] = audio_files[i,:]
        audio_files_buff[i,:]  = buffer(audio_files_norm[i,:],time_shift)
        audio_files_shift1[i,:] = shifter(audio_files_buff[i,:],width_buffer,time_shift,width_no_buffer)
        audio_files_loud[i,:] = louder(audio_files_buff[i,:],vol_shift)
        audio_files_noisy[i,:] =  fast_noise(audio_files_buff[i,:], 0.001, width_buffer)

    n_augments = 4
        
    data_for_NN = np.zeros((n_augments,n_samples,width_buffer))
    data_for_NN[0,:,:] = audio_files_buff # All normalised 1st
    data_for_NN[1,:,:] = audio_files_shift1
    data_for_NN[2,:,:] = audio_files_noisy
    data_for_NN[3,:,:] = audio_files_loud
        
    return data_for_NN

def plot_sample(augmented_samples,sample_index):
    for i in range(augmented_samples.shape[0]):
        plt.plot(augmented_samples[i,sample_index,:])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title("Augmented Waveforms: Sample "+str(sample_index))
    plt.savefig("plot_sample",dpi=300)
    plt.show()
    
def hsr_loader(folder_path,df):
    
    n_samples = len(df['file'])
    hsr=22050*2
    
    filenames = os.listdir(folder_path)
    audio_size = 0
    n_samples = len(filenames)
    print('Finding longest')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+filenames[i], sr=hsr)
        if len(audio)>audio_size:
            audio_size = len(audio)
            index_longest = i
    
    audio_size_hsr = int(audio_size*(hsr/22050))
    audio_files_hsr = np.zeros((n_samples,audio_size_hsr))
    print('Loading files')
    for i in tqdm(range(len(df['file']))):
        audio, _ = librosa.load((folder_path+df['file'][i]), sr=hsr)
        padding_amount = int((audio_size_hsr - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) != 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files_hsr[i,:] = audio_padded
    return audio_files_hsr, index_longest, audio_size_hsr

def lsr_loader(folder_path,df,audio_size):
    n_samples = len(df['file'])
    lsr=22050/2
    audio_size_lsr = int(audio_size*(lsr/22050))+1
    audio_files_lsr = np.zeros((n_samples,audio_size_lsr))
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
        print("Augmentation ",(i+1))
        for j in tqdm(range(n_samples)):
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
        print("Augmentation ",(i+1),"/",n_augments)
        for j in tqdm(range(n_samples)):
            audio_mel[i,j,:,:] = np.abs(librosa.feature.melspectrogram(y=augmented_samples[i,j,:]))
    return audio_mel

def spec_plot(spec):
    librosa.display.specshow(spec,x_axis="time",y_axis="linear")
    plt.colorbar()
    plt.show()
    
def listen(sample):
    sf.write('test_audio.wav', sample, 44100, 'PCM_24')
    playsound('test_audio.wav')
    os.remove("test_audio.wav")
    
def listen_mel(sample):
    waveform = librosa.feature.inverse.mel_to_audio(sample,sr=22050)
    sf.write('test_audio.wav', waveform, 22050, 'PCM_24')
    playsound('test_audio.wav')
    os.remove("test_audio.wav")
    
def rand_slices(set_sizes,audio):
    SETS = []
    n_sets = len(set_sizes)
    for i in set_sizes:
        SETS.append(np.array_split(audio,i)) 
    sets_expanded = []
    for i in range(n_sets):
        for j in range(set_sizes[i]):
            sets_expanded.append(SETS[i][j])
    return SETS,sets_expanded,n_sets

def pad_from_list(sample_list,audio_size):
    # audio_size = 0
    n_samples = len(sample_list)
    
    # for j in range(n_samples):
    #     if len(sample_list[j]) > audio_size:
    #         audio_size = len(sample_list[j])
            
    audio_files = np.zeros((n_samples,audio_size))
    
    for i in tqdm(range(n_samples)):
        if len(sample_list[i]) < audio_size:    
            padding_amount = int((audio_size - len(sample_list[i]))/2)
            audio_padded = np.pad(sample_list[i],padding_amount)  
            if len(audio_padded) == audio_size:
                audio_files[i,:] = audio_padded
            else:
                diff = audio_size - len(audio_padded)
                audio_padded = np.append(audio_padded,np.zeros(diff))
                audio_files[i,:] = audio_padded
    return audio_files
    
def add_nzebra(zebra_audio,zebra_labels,nzebra_audio,n_zebra_labels):
    all_audio = np.concatenate((zebra_audio,nzebra_audio), axis=0)
    all_labels = np.append(zebra_labels,n_zebra_labels)
    return all_audio, all_labels

def gen_nzebra(nzebra_path,audio_size):
    audio_nzebra, _ = librosa.load(nzebra_path)
    set_sizes = np.random.randint(100,1000,2)
    _, nzebra_audio, n_sets = rand_slices(set_sizes,audio_nzebra)
    nzebra_audio_pad = pad_from_list(nzebra_audio,audio_size)
    labels_nzebra = np.array(['not_zebra']*len(nzebra_audio_pad))
    return nzebra_audio_pad, labels_nzebra

def save_nzebra(nzebra_audio):
    for i in range(nzebra_audio.shape[0]):
        sf.write('nzebra/nzebera_sample_' + str(i)+'.wav', nzebra_audio[i], 44100, 'PCM_24')
    
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

def balance_stats(audio_files,labels):
    classes = np.unique(labels)
    split_calls = []
    for call in classes:
        call_idx = np.where(labels==call)
        calls = audio_files[call_idx]
        split_calls.append(calls)
    class_rel_pop = np.zeros(len(classes))
    multiplier = np.zeros(len(classes))
    new_multiplier = np.zeros(len(classes))
    balanced = []
    for i in range(len(classes)):
        class_rel_pop[i]=len(split_calls[i])/len(audio_files)
        multiplier[i] = (1/class_rel_pop[i]) - len(classes)
    biggest = np.min(multiplier)
    for j in range(len(multiplier)):
        if multiplier[j] > 0:
            new_multiplier[j] = int(np.abs(multiplier[j]/biggest))
        else:
            new_multiplier[j] = 1
    return np.around(class_rel_pop,3)

def balancer(audio_files,labels):
    print("Initial class proportions:")
    print(balance_stats(audio_files,labels))
    classes = np.unique(labels)
    split_calls = []
    split_labels = []
    for i in range(len(classes)):
        call_idx = np.where(labels==classes[i])
        calls = audio_files[call_idx]
        split_calls.append(calls)
        split_labels.append([classes[i]]*len(split_calls[i]))
    class_rel_pop = np.zeros(len(classes))
    multiplier = np.zeros(len(classes))
    new_multiplier = np.zeros(len(classes))
    for i in range(len(classes)):
        class_rel_pop[i]=len(split_calls[i])/len(audio_files)
        multiplier[i] = (1/class_rel_pop[i]) - len(classes)
    biggest = np.min(multiplier)
    for j in range(len(multiplier)):
        if multiplier[j] > 0:
            new_multiplier[j] = int(np.abs(multiplier[j]/biggest))
        else:
            new_multiplier[j] = 1
    balanced = audio_files
    balanced_labels = np.array(labels)
    for i in range(len(classes)):
        print('----------------------')
        print("Balancing:",classes[i])
        print("Multiplier:",new_multiplier[i])
        for k in range(int(new_multiplier[i]-1)):
            balanced = np.concatenate((balanced, split_calls[i]), axis=0)
            balanced_labels = np.append(balanced_labels,split_labels[i])
    print('----------------------')
    print("Final class proportions:")
    print(balance_stats(balanced,balanced_labels))
    return balanced, balanced_labels

def pad_and_db(audio_list,audio_size):
    n_samples = len(audio_list)
    audio_files = np.zeros((n_samples,audio_size))
    for i in tqdm(range(n_samples)):
        audio = librosa.amplitude_to_db(audio_list[i])
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if (len(audio_padded) % 2) == 0:
            audio_padded = np.append(audio_padded,[0])
        audio_files[i,:] = audio_padded
    return audio_files

def pad(audio_list,audio_size):
    n_samples = len(audio_list)
    audio_files = np.zeros((n_samples,audio_size))
    for i in range(n_samples):
        audio = audio_list[i]
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if len(audio_padded) == audio_size:
            audio_files[i,:] = audio_padded
        else:
            audio_padded = np.append(audio_padded,[0])
            audio_files[i,:] = audio_padded
    return audio_files

def length_finder_bad_labels(folder_path,percentile):
    """ Loads and pads audio files to be of uniform shape"""
    filenames = os.listdir(folder_path)
    labels = gen_labels(filenames)
    n_samples = len(filenames)
    lengths = np.zeros((n_samples))
    idxs = np.arange(0,n_samples,1)
    audio_list = []
    print('Finding file lengths')
    for i in tqdm(range(n_samples)):
        audio, _ = librosa.load(folder_path+filenames[i],sr=22050)
        audio_list.append(audio)
        lengths[i] = len(audio)
    ninetieth_perc = np.percentile(lengths,percentile)
    audio_list_short = []
    labels_short = []
    lengths_short = []
    removed_audio = []
    removed_labels = []
    idxs_short = []
    print("Reducing to "+str(percentile)+"th percentile")
    for i in tqdm(range(n_samples)):
        if len(audio_list[i])<ninetieth_perc:
            audio_list_short.append(audio_list[i])
            labels_short.append(labels[i])
            lengths_short.append(len(audio_list[i]))
            idxs_short.append(idxs[i])
        else:
            removed_audio.append(audio_list[i])
            removed_labels.append(labels[i])      
    max_length = int(max(lengths_short))
    idx_longest = np.argmax(np.array(lengths_short))
    print("Data size reduction: ",np.around(ninetieth_perc/max_length,3))
    print("Removed classes: ",np.unique(removed_labels))
    print("New max length: ",max(lengths_short))
    print("Number of samples removed:",len(removed_labels))
    plt.scatter(idxs,lengths,label="Removed")
    plt.scatter(idxs_short,lengths_short,label=str(percentile)+"th percentile")
    plt.xlabel("Index")
    plt.ylabel("Length")
    plt.legend()
    plt.show()
    return max_length,audio_list_short,labels_short, idx_longest

def pad_bad_lables(audio_list,audio_size):
    n_samples = len(audio_list)
    audio_files = np.zeros((n_samples,audio_size))
    for i in range(n_samples):
        audio = audio_list[i]
        if len(audio)>audio_size:
            diff = int((audio_size-len(audio))/2)
            audio = audio[diff:-diff]
        padding_amount = int((audio_size - len(audio))/2)
        audio_padded = np.pad(audio,padding_amount)
        if len(audio_padded) == audio_size:
            audio_files[i,:] = audio_padded
        else:
            audio_padded = np.append(audio_padded,[0])
            audio_files[i,:] = audio_padded
    return audio_files
