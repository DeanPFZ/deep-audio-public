import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
import time

class Audio_Processor:
    def __init__(self, path):
        self.audio_dir = path

    def set_audio_dir(self, path):
        self.audio_dir = path

    def preprocess(self, file):
        y, sr = librosa.load(self.audio_dir + file)
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

    def preprocess_df(self, data):
        processed_data = np.array([])
        for file in data['filename']:
            if(processed_data.size):
                processed_data = np.vstack((processed_data, preprocess(file)))
            else:
                processed_data = preprocess(file)
        return processed_data

    def preprocess_df_parallel(self, data):
        p = mp.Pool(mp.cpu_count())
        return np.vstack(p.map(preprocess, data['filename']))

    def preprocess_fold(self, fld, data, parallel=False):

        f_df = data[data['fold'] == fld]
        if parallel:
            return preprocess_df_parallel(f_df, audio_dir)
        else:
            return preprocess_df(f_df, audio_dir)