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
    model.add(Dense(38, activation='relu', input_shape=(38,)))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))
    
    model.summary()
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
