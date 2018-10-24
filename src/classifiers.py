from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

def gpu_train(kind, n_components, train_X, train_y = None):
    clas = None
    if kind == 'dnn':
        clas = skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10], n_classes=n_components)
        clas.fit(train_X, train_Y)
        return clas
    else:
        pass
