import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
import time

def preprocess(path):
    y, sr = librosa.load(path)
    # Trim silence from signal
    y, _ = librosa.effects.trim(y)
    # Calculate the first 13 mfcc's
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Get first derivative of the mfccs
    delta = librosa.feature.delta(mfccs)
    # Get second derivative of mfccs
    delta_2 = librosa.feature.delta(mfccs, order=2)
    del y
    del sr
    return np.vstack((mfccs[1:], delta, delta_2)).transpose()

def preprocess_df(data, audio_dir):
    processed_data = np.array([])
    for file in data['filename']:
        full_path = audio_dir + file
        if(processed_data.size):
            processed_data = np.vstack((processed_data, preprocess(full_path)))
        else:
            processed_data = preprocess(full_path)
    return processed_data

def preprocess_df_parallel(data, audio_dir):
    p = mp.Pool(mp.cpu_count())
    return np.vstack(p.map(preprocess, audio_dir + data['filename']))

def preprocess_fold(fld, data, audio_dir, parallel=False):
    
    f_df = data[data['fold'] == fld]
    if parallel:
        return preprocess_df_parallel(f_df, audio_dir)
    else:
        return preprocess_df(f_df, audio_dir)