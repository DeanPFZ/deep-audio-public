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
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def fit(model, X, y, batch_size=128, epochs=20, verbose=0, validation_data=None):
    history = model.fit(X, y, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=verbose, 
                        validation_data=validation_data)
    
def score(model, X, y, verbose=0):
    score = model.evaluate(X, y, verbose)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def build_classifier(epochs=100, batch_size=5):
    return KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=1)

def k_fold_accuracy(estimator, X, y, k=10, shuffle=True):
    kfold = StratifiedKFold(n_splits=k, shuffle=shuffle)
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))