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

    # Returns the standard deviation of the data, the mean of the data, and the noise calculated from mean and stddev
    def std_dev_mean_noise(self, data):
        # Standard deviation of data
        stddev = np.std(data, axis=1)

        mean = np.mean(data, axis=1)

        sig_noise = mean / stddev

        return stddev, mean, sig_noise

        
    def preprocess(self, file):
#         feature_data = []
#         # Read file for librosa
#         data, sample_rate = librosa.load(self.audio_dir + file)

#         # Trim data more than already done
#         trimmed_data, _ = librosa.effects.trim(y=data)

#         # Get Mel Spectrogram
#         S = librosa.feature.melspectrogram(y=trimmed_data, sr=sample_rate, n_fft=512, hop_length=8, fmax=8000)

#         # Get mfcc features
#         mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)

#         attribs = self.std_dev_mean_noise(mfccs[1:])
# #         print(attribs)
#         feature_data = np.hstack(attribs)
        
#         return feature_data

        y, sr = librosa.load(self.audio_dir + file)
        # Trim silence from signal
        y, _ = librosa.effects.trim(y)
        # Calculate the first 13 mfcc's
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=8, n_fft=4096)
        # Get first derivative of the mfccs
        delta = librosa.feature.delta(mfccs)
        # Get second derivative of mfccs
        delta_2 = librosa.feature.delta(mfccs, order=2)
        # Get rid of zero vectors
        return np.vstack((mfccs[1:], delta, delta_2)).transpose()

    def preprocess_df(self, data):
        dfs = []
        for index, sample in data.iterrows():
            tmp = pd.DataFrame(self.preprocess(sample.filename))
            tmp['target'] = sample['target']
            dfs.append(tmp)
        return pd.concat(dfs)

    def preprocess_df_parallel(self, data):
        p = mp.Pool(mp.cpu_count())
        df = pd.DataFrame()
        for target in data.target.unique():
            tmp = pd.DataFrame(np.vstack(p.map(self.preprocess, data['filename'])))
            tmp['target'] = target
            df.append(tmp, ignore_index=True)
        return df

    def preprocess_fold(self, fld, data, parallel=False):
        f_df = data[data['fold'] == fld]
        if parallel:
            return self.preprocess_df_parallel(f_df)
        else:
            return self.preprocess_df(f_df)