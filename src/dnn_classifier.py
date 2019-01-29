from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_baseline():
    # Create Model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(38,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))
    
    model.summary()
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
