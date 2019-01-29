from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.training_utils import multi_gpu_model
from keras.utils import to_categorical
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise

def create_baseline():
    # Create Model
    model = Sequential()
    model.add(Dense(38, activation='relu', input_shape=(38,)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model

def multiclass_shallow_net():
    # Create Model
    model = Sequential()
    model.add(Dense(38, activation='relu', input_shape=(38,)))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='softmax'))
    
    model.summary()
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model

def create_mfcc_baseline():
    # 6 channels (!), maybe 1-sec audio signal, for an example.
    input_shape = (1, 44100)
    sr = 44100
    model = Sequential()
    # A mel-spectrogram layer
    model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=input_shape, 
              return_decibel_spectrogram=True, power_spectrogram=2.0, 
              trainable_kernel=False, name='static_stft'))
    model.add(Convolution2D(32, (3, 3), name='conv1', activation='relu'))
    model.add(MaxPooling2D((25, 17)))
    model.add(Convolution2D(32, (10, 10), name='conv2', activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])    
    # Compile the model
    model.compile('adam', 'categorical_crossentropy') # if single-label classification
    
    return model