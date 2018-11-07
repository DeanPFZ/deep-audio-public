from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.training_utils import multi_gpu_model
from keras.utils import to_categorical

def create_baseline():
    # Create Model
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(38,)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model
