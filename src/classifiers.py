from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow.contrib.learn as skflow

def cpu_train(kind, n_components, train_X, train_y = None):
    clas = None
    if kind == 'gmm':
        clas = GaussianMixture(n_components=n_components, verbose=1)
    elif kind == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_components)
    elif kind == 'rf':
        clas = RandomForestClassifier(n_estimators=n_components)
    elif kind == 'svm':
        clas = SVC(C=0.1)
    else:
        pass
    clas.fit(X=train_X, y=train_y)
    return clas

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)

def gpu_train(kind, n_components, train_X, train_y = None):
    clas = None
    if kind == 'dnn':
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_X.shape[0])]
        clas = skflow.DNNClassifier(hidden_units=[128,64,32], n_classes=n_components, feature_columns=feature_columns)
        train_steps=1000
        batch=1000
        clas.fit(train_X, train_y, batch_size=batch, steps=train_steps)
        return clas
    else:
        pass
