__author__ = "j.logas"

from numpy.random import seed
from seaborn import load_dataset
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

def moons_data():
    seed(0)
    X, y = make_moons(200, noise=0.20)
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    return df

# Gets pandas dataframe of iris dataset
def iris_data():
    return load_dataset('iris')

def planets_data():
    return load_dataset('planets')

def tips_data():
    return load_dataset('tips')

def titanic_data():
    return load_dataset('titanic')

def flights_data():
    return load_dataset('flights')

# Splits the dataframe into a training and testing set with default 20% split
def split_training_test(df, test_size=0.2):
    return train_test_split(df, test_size=test_size)

# Enumerates strings in dataframe returns the label encoder that can reverse transform
def enumerate_strings(df, exclude):
    column_enums = {}
    for column in df.columns:
        # If the column is made up of strings convert
        if not pd.api.types.is_numeric_dtype(df[column]) and column not in exclude:
            le = preprocessing.LabelEncoder()
            le.fit(df[column])
            df[column] = le.transform(df[column])
            column_enums[column] = le
    return column_enums

# Normalizes given X and returns a new X
def normalize_data(df, target, scalar=None):
    if scalar == None:
        scaler = preprocessing.StandardScaler()
        scaler.fit(df.loc[:, df.columns != target])
    df.loc[:,df.columns != target] = scaler.transform(df.loc[:,df.columns != target])
    return df, scalar

# Supersample of data with equal distribution
def balanced_supersample(x,y):
    y = pd.Series(y)
    x = pd.DataFrame(x)
    y_out = pd.DataFrame(y.values)
    x_out = pd.DataFrame(x)
    counts = pd.Series(y).value_counts()
    counts -= counts.iloc[0]
    for item in counts.iteritems():
        if item[1] == 0:
            continue
        else:
            samples = y[y == item[0]].sample(-1 * item[1], replace=True)
            y_out = pd.concat([y_out, samples], ignore_index=True)
            x_out = pd.concat([x_out, x.loc[samples.index]], ignore_index=True)
    return x_out, y_out.squeeze()
    
# Subsample of data with equal distribution
def balanced_subsample(x,y):
    y = pd.Series(y)
    x = pd.DataFrame(x)
    y_out = pd.DataFrame()
    x_out = pd.DataFrame()
    counts = pd.Series(y).value_counts(ascending=True)
    min_count = counts.iloc[0]
    for item in counts.iteritems():
        if item[1] == min_count:
            y_out = pd.concat([y_out, y[y == item[0]]])
            x_out = pd.concat([x_out, x.loc[y_out.index]], ignore_index=True)
        else:
            samples = y[y == item[0]].sample(min_count)
            y_out = pd.concat([y_out, samples], ignore_index=True)
            x_out = pd.concat([x_out, x.loc[samples.index]], ignore_index=True)
    return x_out, y_out.squeeze()
            