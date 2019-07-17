
import numpy as np
import pandas as pd
import inspect
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, InputLayer
from kapre.time_frequency import Melspectrogram, Spectrogram

from keras.wrappers.scikit_learn import KerasClassifier

SR=16000

class Mixed_Multilayer(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=128, validation_split=0.05,
                       a_epochs=50, a_batch_size=128,
                       i_epochs=50, i_batch_size=128,
                       verbose=1, proc=None):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            pass
    
    
    def fit(self, X, y):
        
        X_b = np.array([i for i in X['base']])
        X_d = pd.concat([j for j in X['derived']], ignore_index=True)
        
        y_p = []
        yy = []
        a_y = []
        i_y = []
        for i in range(0, len(X)):
            t_y = y.at[i, 'h_target']
            t_yy = y.at[i, 'target']
            length = len(X.at[i, 'derived'])
            yy += [t_yy] * length
            y_p += [t_y] * length
            if t_y == 0:
                a_y += [t_yy] * length
            else:
                i_y += [t_yy] * length
        
        if self.proc:
            X_d = self.proc.fit_transform(X_d, yy)
        
        dims = X_d.shape[1]
#         print(X_b.shape)
#         print(X_d.shape)
#         print(np.array(yy).shape)
#         print(np.array(y_p).shape)
        
        """Top layer of hierarchy"""
        self.clf = KerasClassifier(build_fn=self.deep_net,
                                   feature_count=SR * 5,
                                   epochs=self.epochs, 
                                   batch_size=self.batch_size, 
                                   validation_split=self.validation_split,
                                   verbose=self.verbose
                                  )
        self.clf.fit(X_b[:, np.newaxis, :], y.h_target)
        
        """Animal Layer"""
        self.a_clf = KerasClassifier(build_fn=self.deep_net_a,
                                       feature_count=dims,
                                       epochs=self.a_epochs, 
                                       batch_size=self.a_batch_size, 
                                       validation_split=self.validation_split,
                                       verbose=self.verbose
                                    )
        
        self.a_clf.fit(X_d[np.isin(y_p, 0)], a_y)
        
        """Interacting Materials Layer"""
        self.i_clf = KerasClassifier(build_fn=self.deep_net_i, 
                                       feature_count=dims,
                                       epochs=self.i_epochs, 
                                       batch_size=self.i_batch_size, 
                                       validation_split=self.validation_split,
                                       verbose=self.verbose
                                    )
        self.i_clf.fit(X_d[np.isin(y_p, 1)], i_y)
        
        return self
        
    def predict(self, X, y=None):
        predictions=[]

        X_b = np.array([i for i in X['base']])
        X_d = X['derived']
        
        prob = self.clf.predict_proba(X_b[:, np.newaxis, :], verbose=0)

        for i, d in enumerate(X_d):
#             print("Document: " + str(i))
            # Transform before estimation
            if self.proc:
                d = self.proc.transform(d)
            
            if prob[i,0] > 0.75:
                pred = self.a_clf.predict(d, verbose=0)
            elif prob[i,1] > 0.75:
                pred = self.i_clf.predict(d, verbose=0)
            else:
                a_prob = self.a_clf.predict_proba(d, verbose=0) * prob[i, 0]
                i_prob = self.i_clf.predict_proba(d, verbose=0) * prob[i, 1]

                a_prob = np.prod(a_prob, axis=0)
                i_prob = np.prod(i_prob, axis=0)
                
                if(np.max(a_prob) > np.max(i_prob)):
                    pred = self.a_clf.predict(d, verbose=0)
                else:
                    pred = self.i_clf.predict(d, verbose=0)

            pred = np.bincount(pred).argmax()
            predictions.append(pred)
        
        return predictions
    
    def predict_per_frame(self, X, y=None):
        predictions=[]

        X_b = np.array([i for i in X['base']])
        X_d = X['derived']
        
        prob = self.clf.predict_proba(X_b[:, np.newaxis, :], verbose=0)

        for i, d in enumerate(X_d):
#             print("Document: " + str(i))
            # Transform before estimation
            if self.proc:
                d = self.proc.transform(d)
            
            if prob[i,0] > 0.75:
                pred = self.a_clf.predict(d, verbose=0)
            elif prob[i,1] > 0.75:
                pred = self.i_clf.predict(d, verbose=0)
            else:
                a_prob = self.a_clf.predict_proba(d, verbose=0) * prob[i, 0]
                i_prob = self.i_clf.predict_proba(d, verbose=0) * prob[i, 1]

                a_prob = np.prod(a_prob, axis=0)
                i_prob = np.prod(i_prob, axis=0)
                
                if(np.max(a_prob) > np.max(i_prob)):
                    pred = self.a_clf.predict(d, verbose=0)
                else:
                    pred = self.i_clf.predict(d, verbose=0)

            predictions.append(pred)
        
        return predictions
    
    def predict_proba(self, X, y=None):
        
        X_b = np.array([i for i in X['base']])
        X_d = X['derived']
        
        prob = self.clf.predict_proba(X_b[:, np.newaxis, :], verbose=0)
        
        for i, d in enumerate(X_d):
            if self.proc:
                X_d = self.proc.transform(d)
                
            prob_a = np.multiply(np.prod(self.a_clf.predict_proba(X_d, verbose=0), axis=0),prob[i,0])
#             print(prob_a.shape)

            prob_i = np.multiply(np.prod(self.i_clf.predict_proba(X_d, verbose=0), axis=0),prob[i,0])
#             print(prob_i.shape)
#             print()
        
        probs = [None] * 50
        for counter, j in enumerate(self.a_clf.classes_):
            probs[j] = prob_a[counter]
        for counter, j in enumerate(self.i_clf.classes_):
            probs[j] = prob_i[counter]
            
        return np.array(probs)
        
    
    def deep_net(self, feature_count):
        # Create Model
        model = Sequential()
        model.add(Melspectrogram(
            sr=SR,
            n_mels=128,
            power_melgram=1.0,
            input_shape=(1, feature_count),
            trainable_fb=False,
            fmin = 800,
            fmax = 8000
        ))
        model.add(Convolution2D(32, 9, 9, name='conv1', activation='relu'))
        model.add(MaxPooling2D((25, 17)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()


        return model

    def deep_net_a(self, feature_count):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(50,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(18, kernel_initializer='normal', activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()


        return model

    def deep_net_i(self, feature_count):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(50,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, kernel_initializer='normal', activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()


        return model
