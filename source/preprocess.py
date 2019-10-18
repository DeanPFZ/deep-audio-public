import pandas as pd
import numpy as np
import pickle
import soundfile as sf
import os

import keras as krs
import kapre as kpr

class Audio_Processor:

    def __init__(self, path, sr=44100):
        self._audio_dir = path
        self._sr = sr

    def save_obj(self, obj, name):
        if not os.path.exists(self._audio_dir + '/preprocessed_objs/'):
            os.makedirs(self._audio_dir + '/preprocessed_objs/')
        obj.to_csv(self._audio_dir + '/preprocessed_objs/' + name + '.csv', index=None, sep=',')

    def load_obj(self, name):
        return pd.read_csv(self._audio_dir + '/preprocessed_objs/' + name + '.csv')

    def set_audio_dir(self, path):
        self._audio_dir = path

    def set_audio_sample_rate(self, sr):
        self._sr = sr

    def __mel_spec_model(self, input_shape, n_mels, power_melgram, decibel_gram):
        model = Sequential()