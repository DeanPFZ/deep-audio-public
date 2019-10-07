import pandas as pd
from librosa import display, feature, power_to_db
import librosa
import numpy as np
import time
import pickle
import soundfile as sf
import os

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

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, scale
from sklearn.cluster import MiniBatchKMeans

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



def lowpass(y, N=3,low_pass_freq,sr):
    #for digital filters low-pass frequency in rads is given by lowpass/(sr/2)
    Wn = (2*low_pass_freq)/sr
    # First, design the Buterworth 
    B, A = signal.butter(N, Wn, output='ba')
    return signal.filtfilt(B,A, y)
   
class Audio_Processor:
   
    def __init__(self, path, sr=16000, debug=False):
        self._audio_dir = path
        self._sr = sr
        self._mfcc_model = None
        self._debug = debug

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
        
    def __check_model(self, model):
        if self._debug:
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
        if self._debug:
            print("S shape: " + str(S.shape))
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
#                 print(tmp_y.shape)
                out_y = np.append(out_y, tmp_y, axis=1)
            elif samplen == 0:
                continue
            elif samplen < blocksize:
                diff = blocksize-samplen
                padded = np.pad(y[i], 
                                (int(np.ceil(diff/2)),int(np.floor(diff/2))),
                                'edge')[:, np.newaxis]
#                 print(padded.shape)
                out_y = np.append(out_y, padded, axis=1)
            else:
#                 print((y[i])[:,np.newaxis].shape)
                out_y = np.append(out_y, (y[i])[:,np.newaxis], axis=1)
        return out_y.T[1:]
    
    def load_file(self, filepath, blocksize=1024, overlap=512):
        items = []
        if self._debug:
            print("File Processing", end="", flush=True)
        sr = sf.info(filepath).samplerate
        if(sr != self._sr):
            blocksize = int(sr / (self._sr/blocksize))
            if overlap > 0:
                overlap = int(sr / (self._sr/overlap))
        blockgen = sf.blocks(filepath,
                             blocksize=blocksize,
                             overlap=overlap,
                             always_2d=True,
                             fill_value=0.0)
        for bl in blockgen:
            if not np.any(bl):
                continue
            if self._debug:
                print(".", end="", flush=True)
            y = bl.transpose()
            y = librosa.resample(y, sr, self._sr)
            # Lowpass
            y = lowpass(y)
            y = y[:int(blocksize)]
            y = y[np.newaxis, :]
            items.append(y)

        if self._debug:
            print("Done")

        return np.vstack(items)
    
    def load_audio(self, f_df, blocksize, overlap):
        yy = []
        for i, sample in f_df.iterrows():
            yy.append(self.load_file(self._audio_dir + sample.filename, blocksize, overlap))
            if self._debug:
                print(yy[-1].shape)
        
        return yy
               
    
    def __preprocess_df(self, data, kind, blocksize, overlap, n_mels, power_melgram, decibel_gram):
        dfs = []
        
        # Load all data
        f_df = data

        f_df.reset_index(inplace=True, drop=True)
            
        # Load the data
        yy = self.load_audio(f_df, blocksize, overlap)
        
        if self._debug:
            print(yy[0])

        if kind == 'mfcc':
            self._mfcc_model = self.__mel_spec_model(
                yy[0][0].shape,
                n_mels,
                power_melgram,
                decibel_gram
            )
            self.__check_model(self._mfcc_model)
            mels = []
            for i in range(0, len(yy)):
                mels.append(self.__evaluate_model(self._mfcc_model, yy[i]))
                
            f_df['metadata'] = None
            if self._debug:
                print("Mels Length: " + str(len(mels)))
            for (i, mel) in enumerate(mels):
                f_df.at[i,'metadata'] = pd.DataFrame(
                    self.__mfcc_encode(mel),
                    columns=feat_cols
                ).replace([np.inf, -np.inf, np.nan], 0)
                
            
            return f_df[['target', 'h_target', 'metadata']]

        elif kind == 'quantized':
            df = self.get_mfccs(data,
                  blocksize=blocksize, 
                  overlap=overlap,
                  n_mels=n_mels,
                  power_melgram=power_melgram,
                  decibel_gram=decibel_gram
                 )
            mfcc = pd.concat(df['mfcc'].values, keys=list(range(len(df))))
            quant = self.quantize_mfccs(mfcc)
            df = pd.DataFrame(quant)
            y_l = pd.Series(data['target'])
            df['target'] = y_l
            return df

    def get_mfccs(self, data, blocksize, overlap, n_mels, power_melgram, decibel_gram):
        dfs = []
        
        # Load all data
        f_df = data

        f_df.reset_index(inplace=True, drop=True)
            
        # Load the data
        yy = self.load_audio(f_df, blocksize, overlap)
        
        if self._debug:
            print(yy[0])

        self._mfcc_model = self.__mel_spec_model(
            yy[0][0].shape,
            n_mels,
            power_melgram,
            decibel_gram
        )
        self.__check_model(self._mfcc_model)
        mels = []
        for i in range(0, len(yy)):
            mels.append(self.__evaluate_model(self._mfcc_model, yy[i]))

        f_df['mfcc'] = None
        if self._debug:
            print("Mels Length: " + str(len(mels)))
        for (i, mel) in enumerate(mels):
            mfccs = []
            for (j, S_j) in enumerate(mel):
                mfccs.append(librosa.feature.mfcc(S=librosa.power_to_db(S_j), n_mfcc=13))
            mfccs = np.hstack(mfccs)
            f_df.at[i,'mfcc'] = pd.DataFrame(mfccs.T).replace([np.inf, -np.inf, np.nan], 0)

        return f_df[['mfcc']]
    
    def quantize_mfccs(self, df, n_clusters=2048):
        scaled = df.apply(scale, raw=True)
        mbk = MiniBatchKMeans(n_clusters=n_clusters,
                              batch_size=n_clusters * 20,
                              max_no_improvement=20,
                              reassignment_ratio=.0001,
                              random_state=42,
                              verbose=True)
        mbk.fit(scaled)
        scaled['label'] = mbk.labels_.tolist()
        sounds = scaled.groupby(level=0)
        acoustic = pd.DataFrame({_id: s.groupby('label').size()
                                for _id, s in sounds}).transpose()
        return acoustic.fillna(0).to_sparse(fill_value=0)
    
    def preprocess_file(self, filename,
                        kind='mfcc',
                        blocksize=16000,
                        overlap=None,
                        n_mels=128,
                        power_melgram=2.0,
                        decibel_gram=True,
                        bag_of_features=True
                       ):
        yy = self.load_file(self._audio_dir + filename, blocksize, overlap)
        if kind == 'mfcc':
            if not self._mfcc_model:
                self._mfcc_model = self.__mel_spec_model(
                    yy[0].shape,
                    n_mels,
                    power_melgram,
                    decibel_gram
                )
                self.__check_model(self._mfcc_model)

            mel = self.__evaluate_model(self._mfcc_model, yy)

            df = pd.DataFrame(
                        self.__mfcc_encode(mel),
                        columns=feat_cols
                   ).replace([np.inf, -np.inf, np.nan], 0)
            
            return df

    
    def preprocess_fold(self, data,
                        kind='mfcc',
                        blocksize=16000,
                        overlap=None,
                        n_mels=128,
                        power_melgram=2.0,
                        decibel_gram=True,
                        balance=None,
                        feature_bag=True,
                        folds=None,
                        random_state=None
                       ):
        try:
            df = self.load_obj(str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))
        except IOError:
            print("Preprocess file not found, building new one")
            start_time = time.time()
            df = self.__preprocess_df(data, kind, blocksize, overlap, n_mels, power_melgram, decibel_gram)
            print("\tBytes: " + str(df.memory_usage(index=True).sum()))
            print("\tProcessing Time: " + str(time.time() - start_time))
            self.save_obj(df, str(kind) + '_' + str(blocksize) + '_' + str(overlap) + 'sr' + str(self._sr))  
        
        if folds:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            df['fold']=None
            i = 1
            for train_index, test_index in skf.split(df, np.zeros(len(df))):
                df.at[test_index, 'fold'] = i
                i += 1
        
        if feature_bag:
            df = self.bag_of_features(df)
            
        return df
            
    def bag_of_features(self, df):
        combined = []
        for i in range(0, len(df)):
            ddf = df.at[i, 'metadata']
            ddf['h_target'] = df.at[i,'h_target']
            ddf['target'] = df.at[i,'target']
            if 'fold' in df.columns:
                ddf['fold'] = df.at[i,'fold']
            combined.append(ddf)

        return pd.concat(combined, ignore_index=True)
            
    def single_vector(self, df):
        combined = []
        for i in range(0, len(df)):
            ddf = df.at[i, 'metadata']
            combined.append(ddf.mean().values)

        combined = pd.DataFrame(np.array(combined), columns=feat_cols)
        combined['h_target'] = df['h_target']
        combined['target'] = df['target']
        combined['fold']=df['fold']
        return combined
                