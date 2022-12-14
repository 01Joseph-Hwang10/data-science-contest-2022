import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def generate_timeseries_expanded():
    """Generates X_timeseries_scales.

    Converts a DataFrame with `['gender', 'age_code', 'region_code', 'c', 't', 's']` columns and additional time column with timestamps to a time series DataFrame.

    Args:
        X (pd.DataFrame): DataFrame. The data should match the following format:
            gender: 1 (if man) | 2 (if woman)
            age_code: Integer
            region_code: Integer
            r"c2022[0-9]{2}": Number of logins. Integer
            r"t2022[0-9]{2}": Number of logins with money transfer. Integer
            r"s2022[0-9]{2}": Duration of logins. Float
    
    Returns:
        np.ndarray: Time series DataFrame. Each column's description (in order):
            gender: 1 (if man) | 2 (if woman)
            age_code: Integer
            region_code: Integer
            c: Number of logins. Integer
            t: Number of logins with money transfer. Integer
            s: Duration of logins. Float
            month_sin: Sine part of encoded month. Float
            month_cos: Cosine part of encoded month. Float
            day_sin: Sine part of encoded day. Float
            day_cos: Cosine part of encoded day. Float
    """

    # Data converters
    dtype = {
        "gender": np.byte,
        "age_code": np.byte,
        "region_code": np.byte,
    }
    for i in range(238):
        dtype["c" + (pd.Timestamp(2022, 1, 1) + pd.Timedelta(days=i)).strftime("%Y%m%d")] = pd.UInt16Dtype()
        dtype["t" + (pd.Timestamp(2022, 1, 1) + pd.Timedelta(days=i)).strftime("%Y%m%d")] = pd.UInt8Dtype()
        dtype["s" + (pd.Timestamp(2022, 1, 1) + pd.Timedelta(days=i)).strftime("%Y%m%d")] = pd.UInt16Dtype()

    # Data Loading
    print("Loading data...")
    # Get pandas timeseries from 2022-01-01 to 2022-08-26
    X_model = pd.read_csv('data/X_model.csv', dtype=dtype)
    # Y_model = pd.read_csv('data/Y_model.csv', dtype={"business": np.bool_})
    datasize = len(X_model)

    scaler = MinMaxScaler()

    # Filling missing values with 0
    print("Filling missing values with 0...")
    X_timeseries_scaled = X_model.fillna(0)

    # Assign id to each row
    print("Assigning id to each row...")
    X_timeseries_scaled['id'] = X_timeseries_scaled.index

    # Convert wide to long
    print("Converting wide to long...")
    X_timeseries_scaled = pd.wide_to_long(X_timeseries_scaled, ["c", "t", "s"], i="id", j="time").reset_index()

    # Perform cyclic date encoding
    # REsource: https://stackoverflow.com/questions/46428870/how-to-handle-date-variable-in-machine-learning-data-pre-processing
    X_timeseries_scaled['time'] = pd.to_datetime(X_timeseries_scaled['time'])
    X_timeseries_scaled['month'] = X_timeseries_scaled['time'].dt.month
    X_timeseries_scaled['month_sin'] = np.sin(X_timeseries_scaled['month']*(2.*np.pi/12))
    X_timeseries_scaled['month_cos'] = np.cos(X_timeseries_scaled['month']*(2.*np.pi/12))
    X_timeseries_scaled['day'] = X_timeseries_scaled['time'].dt.day
    X_timeseries_scaled['day_sin'] = np.sin(X_timeseries_scaled['day']*(2.*np.pi/31))
    X_timeseries_scaled['day_cos'] = np.cos(X_timeseries_scaled['day']*(2.*np.pi/31))
    X_timeseries_scaled = X_timeseries_scaled.drop(['month', 'day', 'time', 'id'], axis=1)

    # Scaling
    print("Scaling...")
    X_timeseries_scaled[['c', 't', 's']] = scaler.fit_transform(X_timeseries_scaled[['c', 't', 's']])

    days = 238

    # Expand time series to 3D array
    print("Expanding time series to 3D array...")
    X_timeseries_expanded = []

    for i in tqdm(range(datasize)):
        X_timeseries_expanded.append(
            [X_timeseries_scaled.loc[i + day * datasize,:].values for day in range(days)]
        )

    return np.array(X_timeseries_expanded)

np.save('data/X_timeseries_expanded.npy', generate_timeseries_expanded())
