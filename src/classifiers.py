import ggmm.gpu as gpuGMM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
    if kind == 'gmm':
        gpuGMM.init(max_ones=train_X.nbytes)
        clas = gpuGMM.GMM(n_components=n_components, n_dimensions=train_X.shape[1])
        clas.fit(X=train_X, n_init=5)
        gpuGMM.shutdown()
        return clas
    else:
        pass
    
def gpu_init(max_ones=(1024*256)):
    gpuGMM.init(max_ones=max_ones)
    
def gpu_shutdown():
    gpuGMM.shutdown()