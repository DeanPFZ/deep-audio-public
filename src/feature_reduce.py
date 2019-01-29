from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation

def decompose(n_components, X, y=None, kind='PCA'):
    _decomp = None
    if kind == 'PCA':
        _decomp = PCA(n_components=n_components, y=y)
        pass
    elif kind == 'ICA':
        _decomp = FastICA(n_components=n_components, y=y)
        pass
    elif kind == 'LDA':
        _decomp = LatentDirichletAllocation(n_components=n_components, y=y)
        pass
    else:
        print('No acceptable decomposition algorithm was specified')
        return X
    return _decomp.fit_transform(X)