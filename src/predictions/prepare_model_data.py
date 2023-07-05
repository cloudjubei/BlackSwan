from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader
from common.datasets import TimeseriesDataset

def prepareModelDataScaleX(df: pd.DataFrame, column_y: str = "price_close", column_names = ["price_open", "price_close", "price_high", "price_low", "volume"], train_size: float = 0.8, test_size: float = 0.1):

    columns = df[column_names].fillna(0).to_numpy(dtype=float)[:-1]
    labels = df[column_y].to_numpy(dtype=float)[1:]

    train_len = int(len(columns) * train_size)
    test_len = int(len(columns) * (train_size + test_size))

    X_train, y_train = columns[:train_len], labels[:train_len]
    X_test, y_test = columns[train_len:test_len], labels[train_len:test_len]
    X_val, y_val = columns[test_len:], labels[test_len:]

    ss = StandardScaler()

    X_train_t = ss.fit_transform(X_train)
    X_test_t = ss.transform(X_test)
    X_val_t = ss.transform(X_val)

    return {
        "X_train": X_train_t,
        "y_train": y_train,
        "X_test": X_test_t,
        "y_test": y_test,
        "X_val": X_val_t,
        "y_val": y_val,
        "ss": ss,
    }

def prepareModelData(df: pd.DataFrame, column_y: str = "price_close", column_names = ["price_open", "price_close", "price_high", "price_low", "volume"], train_size: float = 0.8, test_size: float = 0.1):

    columns = df[column_names].fillna(0).to_numpy(dtype=float)[:-1]
    labels = df[column_y].to_numpy(dtype=float)[1:]

    train_len = int(len(columns) * train_size)
    test_len = int(len(columns) * (train_size + test_size))

    X_train, y_train = columns[:train_len], labels[:train_len]
    X_test, y_test = columns[train_len:test_len], labels[train_len:test_len]
    X_val, y_val = columns[test_len:], labels[test_len:]

    ss = StandardScaler()
    ss_y = StandardScaler()

    X_train_t, y_train_t = ss.fit_transform(X_train), ss_y.fit_transform(y_train.reshape(-1, 1)).squeeze(1)
    X_test_t, y_test_t = ss.transform(X_test), ss_y.transform(y_test.reshape(-1, 1)).squeeze(1)
    X_val_t, y_val_t = ss.transform(X_val), ss_y.transform(y_val.reshape(-1, 1)).squeeze(1)

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "ss": ss,
        "ss_y": ss_y,
    }

def prepareDataLoaders(model_data):
    
    train_dataset = TimeseriesDataset(model_data["X_train"], model_data["y_train"])
    test_dataset = TimeseriesDataset(model_data["X_test"], model_data["y_test"])
    val_dataset = TimeseriesDataset(model_data["X_val"], model_data["y_val"])

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dl, test_dl, val_dl