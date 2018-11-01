from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import time

def cpu_train(kind, n_components, train_X, train_y = None):
    clas = None
    start_time = time.time()
    if kind == 'gmm':
        clas = BayesianGaussianMixture(n_components=n_components, verbose=1)
    elif kind == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_components)
    elif kind == 'rf':
        clas = RandomForestClassifier(n_estimators=n_components)
    elif kind == 'svm':
        clas = SVC(C=0.1)
    elif kind == 'snn':
        clas = MLPClassifier(hidden_layer_sizes=(12, 12, 12))
    else:
        pass
    clas.fit(X=train_X, y=train_y)
    print("\tProcessing Time: " + str(time.time() - start_time))
    return clas

def cpu_train_gmm_preinitialize(gmm, train_X, train_y = None):
    clas = clone(gmm)
    clas.fit(train_X, train_y)
    return clas

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)

def gpu_train(kind, n_components, train_X, train_y = None, train_steps=1000, batch=None):
    clas = None
    start_time = time.time()
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_X.shape[0])]
    if kind == 'dnn':
        clas = skflow.DNNClassifier(hidden_units=[128,64,32], n_classes=n_components, feature_columns=feature_columns)
    if kind == 'snn':
        clas = skflow.DNNClassifier(hidden_units=[12,12,12], n_classes=n_components, feature_columns=feature_columns)
    else:
        pass
    clas.fit(train_X, train_y, batch_size=batch, steps=train_steps)
    print("\tProcessing Time: " + str(time.time() - start_time))
    return clas