#!/usr/bin/env pthon3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    '''preprocess the data'''
    # Load the dataset
    file_path = '/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    df = pd.read_csv(file_path)

    # Drop missing values
    df = df.dropna()

    # Consider only relevant features (open, high, low, close, volume)
    df = df[['Close']]

    # Normalize data
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)

    # Turn the dataset into sequences with a given time window
    window_size = 24
    sequences = []
    targets = []

    for i in range(len(df_normalized) - window_size):
        sequences.append(df_normalized[i:i+window_size])
        targets.append(df_normalized[i+window_size])

    # Make into numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # Save data
    np.save('X.npy', X)
    np.save('y.npy', y)

preprocess_data()

