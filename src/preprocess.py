import pandas as pd
from librosa import display, feature
import numpy as np
import multiprocessing as mp
import time
import pickle
import soundfile as sf

#Relevant Keras/Kapre includes
import keras
import kapre
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.filterbank import Filterbank

# Relevant Wavenet includes
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

def save_obj(obj, name ):
    with open('../preprocessed_objs/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('../preprocessed_objs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
class Audio_Processor:
   
    def __init__(self, path, sr=44100):
        self.audio_dir = path
        self.sr = sr        
        
    def set_audio_dir(self, path):
        self.audio_dir = path
        
    def set_audio_sample_rate(self, sr):
        self.sr = sr

    def __mel_spec_model(self, input_shape, n_mels, power_melgram, decibel_gram):
        model = Sequential()
        model.add(Melspectrogram(
            sr=self.sr,
            n_mels=n_mels,
            power_melgram=power_melgram,
            return_decibel_melgram = decibel_gram,
            input_shape=input_shape,
            trainable_fb=False
        ))
        return model
        
    def __spec_model(self, input_shape, decibel_gram):
        model = Sequential()
        model.add(Spectrogram(
            return_decibel_spectrogram = decibel_gram,
            input_shape=input_shape
        ))
        return model
        
    def __check_model(self, model):
        model.summary(line_length=80, positions=[.33, .65, .8, 1.])

        batch_input_shape = (2,) + model.input_shape[1:]
        batch_output_shape = (2,) + model.output_shape[1:]
        model.compile('sgd', 'mse')
        model.fit(np.random.uniform(size=batch_input_shape), np.random.uniform(size=batch_output_shape), epochs=1)

    def __visualise_model(self, model, src, logam=False):
        n_ch, nsp_src = model.input_shape[1:]
        print(src.shape)
        src = src[:nsp_src]
        src_batch = src[np.newaxis, :]
        pred = model.predict(x=src_batch)
        if keras.backend.image_data_format == 'channels_first':
            result = pred[0, 0]
        else:
            result = pred[0, :, :, 0]
        display.specshow(result, y_axis='linear', fmin=800, fmax=8000, sr=self.sr)
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

        return stddev, mean, sig_noise
    
    def __wavenet_encode(file_path):

        # Load the model weights.
        checkpoint_path = './wavenet-ckpt/model.ckpt-200000'

        # Load and downsample the audio.
        neural_sample_rate = 16000
        audio = utils.load_audio(self.audio_dir + file_path, 
                                 sample_length=400000, 
                                 sr=neural_sample_rate)

        # Pass the audio through the first half of the autoencoder,
        # to get a list of latent variables that describe the sound.
        # Note that it would be quicker to pass a batch of audio
        # to fastgen. 
        encoding = fastgen.encode(audio, checkpoint_path, len(audio))

        # Reshape to a single sound.
        return encoding.reshape((-1, 16))
    
    def __mfcc_encode(self, mel_spec, spec):
        # Calculate the first 13 mfcc's
        mfccs = feature.mfcc(S=mel_spec, n_mfcc=13)
        # Get first derivative of the mfccs
        delta = feature.delta(mfccs)
        # Get second derivative of mfccs
        delta_2 = feature.delta(mfccs, order=2)
        return np.vstack((mfccs[1:], delta, delta_2)).transpose()
    
    def __load_audio(data, fld=None, blocksize=None, overlap=None, debug=False):
        start_time = time.time()
        
        # Load fold data or all data
        if fld:
            f_df = data[data['fold'] == fld]
        else:
            f_df = data
        items = []
        h_cat = []
        cat = []
        for i, sample in f_df.iterrows():
            # Check if blocksize is set, if not load entire file
            if blocksize:
                # Create iterable object to pull in audio samples
                blockgen = sf.blocks(self.audio_dir + sample.filename, 
                                     blocksize=blocksize, 
                                     overlap=overlap, 
                                     always_2d=True, 
                                     samplerate=self.sr,
                                     fill_value=0.0)
                # Iterate over blocks, adding pertinent information for training
                for bl in blockgen:
                    # Ignore blocks that are silent
                    if not np.any(bl):
                        continue
                    y = bl.transpose()
                    y = y[:int(blocksize)]
                    y = y[np.newaxis, :]
                    items.append(y)
                    h_cat.append(sample.h_category)
                    cat.append(sample.target)
            # If not given, load entire audio document
            else:
                y, sr = sf.read(self.audio_dir + sample.filename, 
                                fill_value=0.0,
                                samplerate=self.sr)
                y = y.transpose()
                y = y[np.newaxis, :]
                items.append(y)
                h_cat.append(sample.h_category)
                cat.append(sample.target)
                
        return np.vstack(items), np.array(h_cat), np.array(cat)

    
    def __preprocess_df(self, data, kind, fld, blocksize, overlap, n_mels, power_melgram, decibel_gram):
        dfs = []
        for index, sample in data.iterrows():
            print("Loading Audio")
            loaded_tuple = self.__load_audio(data, fld, blocksize, overlap)
            if kind == 'mfcc':
                # TODO: More intelligently choose input shape (blocksize may be None)
                input_shape=(1,blocksize)
                # Generate keras network to get melgram
                mfcc_model = self.__mel_spec_model(input_shape, n_mels, power_melgram, decibel_gram)
                self.__check_model(mfcc_model)
                # Generate keras network to get spectrogram
                spec_model = self.__spec_model(input_shape, decibel_gram)
                self.__check_model(spec_model)
                
                # Calculate melgram
                melgram = self.__evaluate_model(mfcc_model, loaded_tuple[0])
                # Calculate spectrogram
                specgram = self.__evaluate_model(spec_model, loaded_tuple[0])

                preproc_dat = []
                for i in range(0, spec.shape[0]):
                    preproc_dat.append(__mfcc_encode(melgram[i], specgram[i]))
                return pd.DataFrame(preproc_dat)
            else:
                pass
        return pd.DataFrame()


    def preprocess_fold(self, data,
                        kind='mfcc',
                        fld=None,
                        blocksize=None,
                        overlap=None,
                        n_mels=128,
                        power_melgram=2.0,
                        decibel_gram=True):
        try:
            df = load_obj('fold_' + str(kind) + '_' + str(fld))
        except IOError:
            print("Preprocess file not found, building new one")
            start_time = time.time()
            df = self.__preprocess_df(data, kind, fld, blocksize, overlap, n_mels, power_melgram, decibel_gram)
            print("\tBytes: " + str(df.memory_usage(index=True).sum()))
            print("\tProcessing Time: " + str(time.time() - start_time))
            save_obj(df, 'fold_' + str(kind) + '_' + str(fld))
        return df