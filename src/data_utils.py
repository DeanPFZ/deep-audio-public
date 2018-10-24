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
def enumerate_strings(df):
    column_enums = {}
    for column in df.columns:
        # If the column is made up of strings convert
        if not pd.api.types.is_numeric_dtype(df[column]):
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
    return df

# Tensorflow specific utilities



def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size cannot be None"
    dataset = dataset.batch(batch_size)

    return dataset