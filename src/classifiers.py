from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def cpu_train(kind, n_components, train_X, train_y = None):
    clas = None
    if kind == 'gmm':
        clas = GaussianMixture(n_components=n_components)
    elif kind == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_components)
    elif kind == 'rf':
        clas = RandomForestClassifier(n_estimators=n_components)
    elif kind == 'svm':
        clas = SVC(C=0.1)
    else:
        pass
    clas.fit(train_X, train_y)
    return clas