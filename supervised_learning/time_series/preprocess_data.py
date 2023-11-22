def preprocess_data():
    '''preprocess the data'''
    # Load the dataset
    file_path = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    df = pd.read_csv(file_path)
    #df.dropna()

    # consider only relevant features (open, high, low, close, volume)
    df = df[['Close', 'Timestamp', 'Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]

    # normalize data
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    df.fillna(0)

    # turn the dataset to sequences with a given time window
    window_size = 24
    sequences = []
    targets = []

    for i in range(len(df_normalized) - window_size):
        sequences.append(df_normalized[i:i+window_size])
        targets.append(df_normalized[i+window_size][3])

    # make to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # save data
    np.save('X.npy', X)
    np.save('y.npy', y)
