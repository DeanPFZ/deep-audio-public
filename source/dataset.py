import os
import numpy as np
import keras as ks
import sklearn as skl
import librosa as lbr
import pandas as pd
import logging

class SoundDataset():
    def __init__(self, opt, preprocess_func=None, seed=None, audio_dir="~/data/ESC-50/"):
        self.base = None
        self.opt = opt
        self.seed = seed
        self.preprocess_funcs = preprocess_func
        self.audio_dir = audio_dir
        self.__load_dataset()
        self.test = None
        self.train = None

    def __len__(self):
        return len(self.base)

    def __load_dataset(self):
        # Get CSV file
        logging.info("Loading CSV from database.")
        self.df = pd.read_csv(self.audio_dir + 'meta/esc50.csv')
        # Load raw audio for dataset
        logging.info("Load raw audio from file.")
        self.__load_audio()
        # Pre-process (default None)
        logging.info("Preprocess loaded audio.")
        self.preprocess_funcs(self.base)

    def get_df(self):
        return self.df

    def get_train(self):
        if self.test:
            return self.test
        pass

    def get_test(self):
        pass

    def __load_audio(self):
        aud = []
        for sample in self.df:
            # Check if blocksize is set, if not load entire file
            aud += lbr.load(self.audio_dir + sample.filename, mono=True)

        self.audio = np.array(aud)
    