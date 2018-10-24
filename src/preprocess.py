import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
import time

col_lables = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCCp_0', 'MFCCp_1', 'MFCCp_2', 'MFCCp_3', 'MFCCp_4', 'MFCCp_5', 'MFCCp_6', 'MFCCp_7', 'MFCCp_8', 'MFCCp_9', 'MFCCp_10', 'MFCCp_11', 'MFCCp_12', 'MFCCp2_0', 'MFCCp2_1', 'MFCCp2_2', 'MFCCp2_3', 'MFCCp2_4', 'MFCCp2_5', 'MFCCp2_6', 'MFCCp2_7', 'MFCCp2_8', 'MFCCp2_9', 'MFCCp2_10', 'MFCCp2_11', 'MFCCp2_12', 'category']

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
        dfs = []
        for index, sample in data.iterrows():
            tmp = pd.DataFrame(self.preprocess(sample.filename))
            tmp['category'] = sample.category
            dfs.append(tmp)
        df = pd.concat(dfs, ignore_index=True)
        df.columns = col_lables
        return df

    def preprocess_df_parallel(self, data):
        p = mp.Pool(mp.cpu_count())
        df = pd.DataFrame()
        for category in data.category.unique():
            tmp = pd.DataFrame(np.vstack(p.map(self.preprocess, data['filename'])))
            tmp['category'] = category
            df.append(tmp, ignore_index=True)
        return df

    def preprocess_fold(self, fld, data, parallel=False):
        f_df = data[data['fold'] == fld]
        if parallel:
            return self.preprocess_df_parallel(f_df)
        else:
            return self.preprocess_df(f_df)