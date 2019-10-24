import librosa as lbr
import numpy as np
import pickle
import os

class Preprocessor:
    def __init__(self, sr=44100):
        self._sr = sr

    # def save_obj(self, obj, name):
    #     if not os.path.exists('preprocessed_objs/'):
    #         os.makedirs('preprocessed_objs/')
    #     obj.to_csv('preprocessed_objs/' + name + '.csv', index=None, sep=',')

    # def load_obj(self, name):
    #     return pd.read_csv('preprocessed_objs/' + name + '.csv')

    # Encoding
    def __std_dev_mean_noise(self, data):
        stddev = np.std(data, axis=1)
        mean = np.mean(data, axis=1)
        sig_noise = mean/stddev
        ret_dat = np.hstack((stddev, mean, sig_noise))
        return ret_dat

    # Get deltas and flatten matrix
    def __2d_encode(self, data):
        # Get first derivative of the data
        delta = lbr.feature.delta(data)
        # Second derivative of the data
        delta_2 = lbr.feature.delta(data, order=2)
        
        data = np.vstack((data[1:], delta, delta_2))
        # Reduce to single feature vector
        return self.__std_dev_mean_noise(data)
