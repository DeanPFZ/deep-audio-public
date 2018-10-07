import pandas as pd
import librosa
import numpy as np

def preprocess(path):
    y, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta_2 = librosa.feature.delta(mfccs, order=2)
    return np.vstack((mfccs[1:], delta, delta_2)).transpose()

def preprocess_df(data, audio_dir):
    processed_data = np.array([])
    for file in data['filename']:
        full_path = audio_dir + file
        if(processed_data.size):
            processed_data = np.vstack((processed_data, preprocess(full_path)))
        else:
            processed_data = preprocess(full_path)

def preprocess_fold(fld, data, audio_dir):
    f_df = data[data['fold'] == fld]
    return preprocess_df(f_df, audio_dir)