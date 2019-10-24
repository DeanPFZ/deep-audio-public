import pandas as pd
import numpy as np
import librosa as lbr
import os
from .preprocessor import Preprocessor

class MFCC(Preprocessor):
    def __init__(self, sr=44100, n_mels=13):
        super(sr)
        self.n_mels=n_mels

    # Encode the given data as the given kind of representation
    def preprocess(self, audio, target):
        # Reshape target to 1D
        target = np.reshape(target, (len(target), 1))

        # dataframe with features
        audio_encode = []

        for clip in audio:
            blk_mfcc = lbr.feature.mfcc(y=clip, 
                                        sr=self._sr, 
                                        n_mfcc=self.n_mels)
            audio_encode += [self.__2d_encode(blk_mfcc)]
        
        return {
            'data': audio_encode,
            'target': target,
            'DESCR': 'Audio data converted to mfccs and flattened'
        }