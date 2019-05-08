import pandas as pd
from librosa import display, feature, power_to_db
import librosa
import numpy as np
import multiprocessing as mp
import time
import pickle
import soundfile as sf
import os

from io import StringIO

from sklearn.preprocessing import normalize

#Relevant Keras/Kapre includes
import keras
import kapre
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.filterbank import Filterbank
from kapre.utils import Normalization2D

# Relevant Wavenet includes
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

# Lowpass Filter
import scipy.signal as signal


feat_cols = [
            'mfcc_2_std', 'mfcc_2_mean', 'mfcc_2_noise',
            'mfcc_3_std', 'mfcc_3_mean', 'mfcc_3_noise',
            'mfcc_4_std', 'mfcc_4_mean', 'mfcc_4_noise',
            'mfcc_5_std', 'mfcc_5_mean', 'mfcc_5_noise',
            'mfcc_6_std', 'mfcc_6_mean', 'mfcc_6_noise',
            'mfcc_7_std', 'mfcc_7_mean', 'mfcc_7_noise',
            'mfcc_8_std', 'mfcc_8_mean', 'mfcc_8_noise',
            'mfcc_9_std', 'mfcc_9_mean', 'mfcc_9_noise',
            'mfcc_10_std', 'mfcc_10_mean', 'mfcc_10_noise',
            'mfcc_11_std', 'mfcc_11_mean', 'mfcc_11_noise',
            'mfcc_12_std', 'mfcc_12_mean', 'mfcc_12_noise',
            'mfcc_13_std', 'mfcc_13_mean', 'mfcc_13_noise',
            'mfcc_1_p_std', 'mfcc_1_p_mean', 'mfcc_1_p_noise',
            'mfcc_2_p_std', 'mfcc_2_p_mean', 'mfcc_2_p_noise',
            'mfcc_3_p_std', 'mfcc_3_p_mean', 'mfcc_3_p_noise',
            'mfcc_4_p_std', 'mfcc_4_p_mean', 'mfcc_4_p_noise',
            'mfcc_5_p_std', 'mfcc_5_p_mean', 'mfcc_5_p_noise',
            'mfcc_6_p_std', 'mfcc_6_p_mean', 'mfcc_6_p_noise',
            'mfcc_7_p_std', 'mfcc_7_p_mean', 'mfcc_7_p_noise',
            'mfcc_8_p_std', 'mfcc_8_p_mean', 'mfcc_8_p_noise',
            'mfcc_9_p_std', 'mfcc_9_p_mean', 'mfcc_9_p_noise',
            'mfcc_10_p_std', 'mfcc_10_p_mean', 'mfcc_10_p_noise',
            'mfcc_11_p_std', 'mfcc_11_p_mean', 'mfcc_11_p_noise',
            'mfcc_12_p_std', 'mfcc_12_p_mean', 'mfcc_12_p_noise',
            'mfcc_13_p_std', 'mfcc_13_p_mean', 'mfcc_13_p_noise',
            'mfcc_1_pp_std', 'mfcc_1_pp_mean', 'mfcc_1_pp_noise',
            'mfcc_2_pp_std', 'mfcc_2_pp_mean', 'mfcc_2_pp_noise',
            'mfcc_3_pp_std', 'mfcc_3_pp_mean', 'mfcc_3_pp_noise',
            'mfcc_4_pp_std', 'mfcc_4_pp_mean', 'mfcc_4_pp_noise',
            'mfcc_5_pp_std', 'mfcc_5_pp_mean', 'mfcc_5_pp_noise',
            'mfcc_6_pp_std', 'mfcc_6_pp_mean', 'mfcc_6_pp_noise',
            'mfcc_7_pp_std', 'mfcc_7_pp_mean', 'mfcc_7_pp_noise',
            'mfcc_8_pp_std', 'mfcc_8_pp_mean', 'mfcc_8_pp_noise',
            'mfcc_9_pp_std', 'mfcc_9_pp_mean', 'mfcc_9_pp_noise',
            'mfcc_10_pp_std', 'mfcc_10_pp_mean', 'mfcc_10_pp_noise',
            'mfcc_11_pp_std', 'mfcc_11_pp_mean', 'mfcc_11_pp_noise',
            'mfcc_12_pp_std', 'mfcc_12_pp_mean', 'mfcc_12_pp_noise',
            'mfcc_13_pp_std', 'mfcc_13_pp_mean', 'mfcc_13_pp_noise',
            'scen_std', 'scen_mean', 'scen_noise',
            'sband_std', 'sband_mean', 'sband_noise',
            'sflat_std', 'sflat_mean', 'sflat_noise',
            'sroll_std', 'sroll_mean', 'sroll_noise',
            'rmse_std', 'rmse_mean', 'rmse_noise',
          ]



def lowpass(y, N=3, Wn=0.5):
    # First, design the Buterworth filter
    N  = 3    # Filter order
    Wn = 0.5 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    return signal.filtfilt(B,A, y)
   
class Audio_Processor:
   
    def __init__(self, path, sr=16000):
        self._audio_dir = path
        self._sr = sr

    def save_obj(self, obj, name ):
        if not os.path.exists(self._audio_dir + '/preprocessed_objs/'):
            os.makedirs(self._audio_dir + '/preprocessed_objs/')
        with open(self._audio_dir + '/preprocessed_objs/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open(self._audio_dir + '/preprocessed_objs/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

        
    def set_audio_dir(self, path):
        self._audio_dir = path
        
    def set_audio_sample_rate(self, sr):
        self._sr = sr

    def __mel_spec_model(self, input_shape, n_mels, power_melgram, decibel_gram):
        model = Sequential()
        model.add(Melspectrogram(
            sr=self._sr,
            n_mels=n_mels,
            power_melgram=power_melgram,
            return_decibel_melgram = decibel_gram,
            input_shape=input_shape,
            trainable_fb=False
        ))
        model.add(Normalization2D(str_axis='freq'))
        return model
        
    def __spec_model(self, input_shape, decibel_gram):
        model = Sequential()
        model.add(Spectrogram(
            return_decibel_spectrogram = decibel_gram,
            input_shape=input_shape
        ))
        model.add(Normalization2D(str_axis='freq'))
        return model
        
    def __check_model(self, model, debug=False):
        if debug:
            model.summary(line_length=80, positions=[.33, .65, .8, 1.])

        batch_input_shape = (2,) + model.input_shape[1:]
        batch_output_shape = (2,) + model.output_shape[1:]
        model.compile('sgd', 'mse')
        model.fit(np.random.uniform(size=batch_input_shape), np.random.uniform(size=batch_output_shape), epochs=1)

    def __visualise_model(self, model, src, logam=False):
        n_ch, nsp_src = model.input_shape[1:]
        # print(src.shape)
        src = src[:nsp_src]
        src_batch = src[np.newaxis, :]
        pred = model.predict(x=src_batch)
        if keras.backend.image_data_format == 'channels_first':
            result = pred[0, 0]
        else:
            result = pred[0, :, :, 0]
        display.specshow(result, y_axis='linear', fmin=800, fmax=8000, sr=self._sr)
        plt.show()

    def __evaluate_model(self, model, c_data):
        pred = model.predict(x=c_data)
        if keras.backend.image_data_format == 'channels_first':
            result = pred[0, 0]
        else:
            result = pred[:, :, :, 0]
#         result = np.swapaxes(result, 1, 2)
#         print(result.shape)
        return result
        
    # Returns the standard deviation of the data, the mean of the data, and the noise calculated from mean and stddev
    def __std_dev_mean_noise(self, data):
        # Standard deviation of data
        stddev = np.std(data, axis=1)

        mean = np.mean(data, axis=1)

        sig_noise = mean / stddev

        ret_dat = np.hstack((stddev, mean, sig_noise))

        return ret_dat
    
    def __wavenet_encode(self, audio):

        # Load the model weights.
        checkpoint_path = '../wavenet-ckpt/model.ckpt-200000'

        # Load and downsample the audio.
        # neural_sample_rate = 16000
        # audio = utils.load_audio(self._audio_dir + file_path, 
        #                          sample_length=400000, 
        #                          sr=neural_sample_rate)

        # Pass the audio through the first half of the autoencoder,
        # to get a list of latent variables that describe the sound.
        # Note that it would be quicker to pass a batch of audio
        # to fastgen. 
        audio = np.squeeze(audio)
        # print(audio.shape)
        # print(len(audio))
        if(len(audio.shape) > 1):
            encoding = fastgen.encode(audio, checkpoint_path, audio.shape[1])
        else:
            encoding = fastgen.encode(audio, checkpoint_path, len(audio))

#         print("Pre: " + str(encoding.shape))
        encoding = self.__std_dev_mean_noise(encoding)
#         print("Post: " + str(encoding.shape))
        # Reshape to a single sound.
        return encoding
    
    def __mfcc_encode(self, f):
        items=[]
        S, phase = librosa.magphase(f)
        for (i, S_i) in enumerate(f):
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S_i), n_mfcc=13)
            # First derivatives
            delta = librosa.feature.delta(mfccs)
            # Second Deriv
            delta_2 = librosa.feature.delta(mfccs, order=2)
            # Spectral Centroid
            cent = librosa.feature.spectral_centroid(S=S[i])
            # Spectral Bandwidth
            band = librosa.feature.spectral_bandwidth(S=S[i])
            # Spectral Flatness
            flat = librosa.feature.spectral_flatness(S=S[i])
            # Spectral rolloff
            roll = librosa.feature.spectral_rolloff(S=S[i])
            # RMSE
            rmse = librosa.feature.rmse(S=S[i])

            dat = np.vstack((mfccs[1:], delta, delta_2, cent, band, flat, roll, rmse))

            # Reduce to single feature row
            dat = self.__std_dev_mean_noise(dat)
#             dat = np.hstack((
#                 target,
#                 h_target,
#                 dat
#             ))
            items.append(dat)

        return np.array(items)    

    def __tokenize(self, y, blocksize, overlap):
        y = np.array([y[i[0]:i[1]] for i in librosa.effects.split(y, frame_length=512, hop_length=256, top_db=30)])

        out_y = np.zeros((blocksize,1))

        for i in range(0,len(y)):
            samplen = len(y[i])
            if samplen > blocksize:
                tmp_y = librosa.util.frame(np.ascontiguousarray(y[i]),
                                           frame_length=blocksize,
                                           hop_length=overlap)
                
                out_y = np.append(out_y, tmp_y, axis=1)
            elif samplen < blocksize:
                diff = blocksize-samplen
                padded = np.pad(y[i], 
                                (int(np.ceil(diff/2)),int(np.floor(diff/2))),
                                'edge')[:, np.newaxis]
                
                out_y = np.append(out_y, padded, axis=1)
            else:
                out_y = np.append(out_y, (y[i])[:,np.newaxis], axis=1)
        return out_y.T[1:]
    
    
    def __load_audio(self, f_df, fld, blocksize, overlap, debug=False):
        start_time = time.time()
        
        yy = []
        for i, sample in f_df.iterrows():
            y, sr = librosa.core.load(self._audio_dir + sample.filename, sr=self._sr, mono=True)
            # Lowpass filter
            y = lowpass(y)
            # Normalize
            y /= np.max(y, axis=0)
            # Half wave rectify
            y = y.clip(min=0)
            # Tokenize
            y = self.__tokenize(y, blocksize, overlap)
            
            yy.append(y[:, np.newaxis, :])
        
        return yy

    def __load_file(self, path, blocksize, overlap, debug=False):
        items = []
        y, sr = librosa.core.load(self._audio_dir + sample.filename, sr=self._sr, mono=True)
        # Lowpass filter
        y = lowpass(y)
        # Normalize
        y /= np.max(y, axis=0)
        # Half wave rectify
        y = y.clip(min=0)
        # Tokenize
        y = self.__tokenize(y, blocksize, overlap)

        yy = y[:, np.newaxis, :]
        
        mfcc_model = self.__mel_spec_model(
            yy[0].shape,
            n_mels,
            power_melgram,
            decibel_gram
        )
        self.__check_model(mfcc_model)
        mel = self.__evaluate_model(mfcc_model, yy)
        feat = self.__mfcc_encode(mel, None, None)[0]

        features = pd.DataFrame(feat, columns=feat_cols)
        features = features.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis=0)
        return features

        
                
    
    def __preprocess_df(self, data, kind, fld, blocksize, overlap, n_mels, power_melgram, decibel_gram):
        dfs = []
        
        # Load fold data or all data
        if fld:
            try:
                f_df = data[data['fold'] == fld]
            except TypeError:
                f_df = data[data['fold'].between(fld[0], fld[-1])]
        else:
            f_df = data

        f_df.reset_index(inplace=True, drop=True)
            
        # Load the data
        yy = self.__load_audio(f_df, fld, blocksize, overlap)

        if kind == 'mfcc':
            mfcc_model = self.__mel_spec_model(
                yy[0][0].shape,
                n_mels,
                power_melgram,
                decibel_gram
            )
            self.__check_model(mfcc_model)
            mels = []
            for i in range(0, len(yy)):
                mels.append(self.__evaluate_model(mfcc_model, yy[i]))
                
            f_df['metadata'] = None
            for (i, mel) in enumerate(mels):
                f_df.at[i,'metadata'] = pd.DataFrame(
                    self.__mfcc_encode(mel),
                    columns=feat_cols
                ).replace([np.inf, -np.inf, np.nan], 0)
        
            return f_df.drop(['take', 'src_file', 'esc10', 'filename'], axis=1)

        elif kind == 'wavnet':
#             preproc_dat = self.__wavenet_encode(loaded_tuple[0])
            preproc_dat = self.__wavenet_encode(loaded_tuple[0][0])
            for i in range(1,len(loaded_tuple[0])):
                preproc_dat = np.vstack((preproc_dat, self.__wavenet_encode(loaded_tuple[0][i])))
            
        # Stack the data with labels
        preproc_dat = np.hstack((preproc_dat, l_target, h_target))

        df = pd.DataFrame(preproc_dat)
        df.rename(columns=dict(zip(df.columns[-2:], ['target', 'h_target'])), inplace=True)
        df['target'] = df['target'].astype(int)
        df['h_target'] = df['h_target'].astype(int)
        df.fillna(0, inplace=True)
        return df

#     def preprocess_file(self, path,
#                         kind='mfcc',
#                         blocksize=44100,
#                         overlap=None,
#                         n_mels=128,
#                         power_melgram=2.0,
#                         decibel_gram=True):
#         dat = self.__load_file(path, blocksize, overlap)
#         if kind == 'mfcc':
#             # TODO: More intelligently choose input shape (blocksize may be None)
#             input_shape=(1,blocksize)
#             # Generate keras network to get melgram
#             mfcc_model = self.__mel_spec_model(input_shape, n_mels, power_melgram, decibel_gram)
#             self.__check_model(mfcc_model)
#             # Generate keras network to get spectrogram
#             spec_model = self.__spec_model(input_shape, decibel_gram)
#             self.__check_model(spec_model)
            
#             # Calculate melgram
#             melgram = self.__evaluate_model(mfcc_model, dat)
#             # Calculate spectrogram
#             specgram = self.__evaluate_model(spec_model, dat)

#             mfcc_dat = self.__mfcc_encode(melgram[0], specgram[0])
#             preproc_dat = np.array(mfcc_dat)
#             for i in range(1, specgram.shape[0]):
#                 mfcc_dat = self.__mfcc_encode(melgram[i], specgram[i])
#                 preproc_dat = np.vstack((preproc_dat, mfcc_dat))

#         elif kind == 'wavnet':
#             preproc_dat = self.__wavenet_encode(loaded_tuple[0][0])
#             for i in range(1,len(loaded_tuple[0])):
#                 preproc_dat = np.vstack((preproc_dat, self.__wavenet_encode(loaded_tuple[0][i])))
                
#         if len(preproc_dat.shape) > 1:
#             df = pd.DataFrame(preproc_dat)
#         else:
#             df = pd.DataFrame(preproc_dat[np.newaxis, :])
#         df.fillna(0, inplace=True)
#         return df


    
    def preprocess_fold(self, data,
                        kind='mfcc',
                        fld=None,
                        blocksize=44100,
                        overlap=None,
                        n_mels=128,
                        power_melgram=2.0,
                        decibel_gram=True,
                        feature_bag=True
                       ):
        try:
            if fld:
                df = self.load_obj('fold_' + str(fld) + '_' + str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))
            else:
                df = self.load_obj(str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))  
        except IOError:
            print("Preprocess file not found, building new one")
            start_time = time.time()
            df = self.__preprocess_df(data, kind, fld, blocksize, overlap, n_mels, power_melgram, decibel_gram)
            print("\tBytes: " + str(df.memory_usage(index=True).sum()))
            print("\tProcessing Time: " + str(time.time() - start_time))
            if fld:
                self.save_obj(df, 'fold_' + str(fld) + '_' + str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))
            else:
                self.save_obj(df, str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))  
        
        if feature_bag:
            return self.bag_of_features(df)
        else:
            return df
    
    def bag_of_features(self, df):
        combined = []
        for i in range(0, len(df)):
            ddf = df.at[i, 'metadata']
            combined.append(ddf.mean().values)

        combined = pd.DataFrame(np.array(combined), columns=feat_cols)
        combined['h_target'] = df['h_target']
        combined['target'] = df['target']
        combined['fold'] = df['fold']
        return combined
