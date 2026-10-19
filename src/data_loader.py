import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    # load file based on extension
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".txt"):
        df = pd.read_csv(path, sep=';', low_memory=False)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")

    return df


def preprocess_data(df, scaler=None):
    # drop first column if it's not numeric
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        df = df.iloc[:, 1:]

    # fix decimal format issues
    df = df.replace(',', '.', regex=True)

    # convert everything to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # remove columns with too many missing values
    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))

    # fill remaining missing values
    df = df.fillna(df.mean())

    # remove columns with no variation
    df = df.loc[:, df.std() != 0]

    # check if valid features remain
    if df.shape[1] == 0:
        raise ValueError("No valid numeric features left after preprocessing!")

    # fit scaler on reference data
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

    # use existing scaler for current data
    else:
        scaled_data = scaler.transform(df)

    return scaled_data, scaler


def split_baseline_stream(data, baseline_ratio=0.6):
    # split data into training (baseline) and future (stream)
    split_index = int(len(data) * baseline_ratio)

    baseline_data = data[:split_index]
    stream_data = data[split_index:]

    return baseline_data, stream_data