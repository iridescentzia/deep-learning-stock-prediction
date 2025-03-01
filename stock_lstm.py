import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import yfinance as yf
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.offline as pyo

pyo.init_notebook_mode()

sp500_list = pd.read_csv('sp500_companies.csv', index_col='Symbol')
sp500_list = sp500_list.drop(['SOLV', 'GEV', 'PXD'])

tickers = ' '.join(sp500_list.index.tolist()).replace('.', '-')
sp500_price = yf.download(tickers=tickers, start='2018-01-01', end='2023-12-30')
sp500_price = sp500_price.dropna(how='all')

sp500_price = sp500_price.stack().swaplevel().sort_index()
sp500_price.columns = [col.lower().replace(' ', '_') for col in sp500_price.columns]
sp500_price.index.names = ['name', 'date']

sp500_price.to_parquet('sp500_price_20180101_20231230.parquet')
sp500_price = pd.read_parquet('sp500_price_20180101_20231230.parquet')

symbols = sp500_price.index.get_level_values('name').unique().tolist()


def calculate_moving_average(df, window=14):
    return df['close'].rolling(window=window).mean()


def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band


def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    psar = np.zeros(len(close))
    psar[0] = close[0]
    uptrend = True
    ep = low[0]
    af = step

    for i in range(1, len(close)):
        if uptrend:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            if low[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = low[i]
                af = step
        else:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            if high[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = high[i]
                af = step
        if uptrend:
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
    return psar


def calculate_roc(df, window=14):
    return df['close'].diff(window) / df['close'].shift(window) * 100


def calculate_mda(y_true, y_pred):
    y_true_diff = np.diff(y_true, axis=0)
    y_pred_diff = np.diff(y_pred, axis=0)

    correct_directions = (y_true_diff * y_pred_diff) > 0
    mda = np.mean(correct_directions)

    return mda


def prepare_data(symbol):
    df = sp500_price.loc[symbol]

    raw_data = deepcopy(df)

    if 'name' in raw_data.columns:
        raw_data = raw_data.drop(['name'], axis=1)

    raw_data['ma'] = calculate_moving_average(raw_data)
    raw_data['rsi'] = calculate_rsi(raw_data)

    upper_band, lower_band = calculate_bollinger_bands(raw_data)
    raw_data['bollinger_upper'] = upper_band
    raw_data['bollinger_lower'] = lower_band

    raw_data['psar'] = calculate_parabolic_sar(raw_data)
    raw_data['roc'] = calculate_roc(raw_data)

    raw_data = raw_data.dropna()

    X_raw_data = raw_data.drop(columns=['close'])
    y_raw_data = raw_data[['close']]

    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    X_raw_data = X_scaler.fit_transform(X_raw_data)
    y_raw_data = y_scaler.fit_transform(y_raw_data)

    def prepare_xy(X_raw_data, y_raw_data, lookback):
        data = list()
        for index in range(len(X_raw_data) - lookback):
            data.append(X_raw_data[index: index + lookback])
        data = np.array(data)
        return data, y_raw_data[lookback:]

    lookback = 10
    X_train, y_train = prepare_xy(X_raw_data, y_raw_data, lookback)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    return X_train, y_train, X_scaler, y_scaler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X_data = X
        self.y_data = y

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        X = self.X_data[index]
        y = self.y_data[index]
        return X, y


class LSTM_N(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p, output_size):
        super().__init__()
        self.sequenceclassifier = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p
        )
        self.fc = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        output, _ = self.sequenceclassifier(x)
        output = output[:, -1, :]
        y = self.fc(output)
        return y


input_size = 11
hidden_size = 32
num_layers = 2
dropout_p = 0
output_size = 1

model = LSTM_N(input_size, hidden_size, num_layers, dropout_p, output_size)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_model(model, early_stop, n_epochs, progress_interval, train_batches, val_batches):
    train_losses, valid_losses, lowest_loss = list(), list(), np.inf

    for epoch in range(n_epochs):
        train_loss, valid_loss = 0, 0

        model.train()
        for x_minibatch, y_minibatch in train_batches:
            y_minibatch_pred = model(x_minibatch)
            loss = loss_func(y_minibatch_pred, y_minibatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_batches)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for x_minibatch, y_minibatch in val_batches:
                y_minibatch_pred = model(x_minibatch)
                loss = loss_func(y_minibatch_pred, y_minibatch)
                valid_loss += loss.item()

        valid_loss = valid_loss / len(val_batches)
        valid_losses.append(valid_loss)

        if valid_losses[-1] < lowest_loss:
            lowest_loss = valid_losses[-1]
            lowest_epoch = epoch
            best_model = deepcopy(model.state_dict())
        else:
            if (early_stop > 0) and lowest_epoch + early_stop < epoch:
                print("Early Stopped", epoch, "epochs")
                break

        if (epoch % progress_interval) == 0:
            print(train_losses[-1], valid_losses[-1], lowest_loss, lowest_epoch, epoch)

    model.load_state_dict(best_model)
    return model, lowest_loss, train_losses, valid_losses


def predict(symbol):
    X_train, y_train, X_scaler, y_scaler = prepare_data(symbol)

    train_rawdata = TensorDataset(X_train, y_train)
    VALIDATION_RATE = 0.2
    train_indices, val_indices = train_test_split(
        range(len(train_rawdata)),
        test_size=VALIDATION_RATE
    )
    train_dataset = Subset(train_rawdata, train_indices)
    validation_dataset = Subset(train_rawdata, val_indices)

    minibatch_size = 128
    train_batches = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    val_batches = DataLoader(validation_dataset, batch_size=minibatch_size, shuffle=True)

    nb_epochs = 50
    progress_interval = 3
    early_stop = 30

    trained_model, lowest_loss, train_losses, valid_losses = train_model(
        model, early_stop, nb_epochs, progress_interval, train_batches, val_batches)
    test_batches = DataLoader(train_rawdata, batch_size=minibatch_size, shuffle=False)
    y_test_pred_list, y_test_list = list(), list()
    trained_model.eval()
    with torch.no_grad():
        for x_minibatch, y_minibatch in test_batches:
            y_minibatch_pred = trained_model(x_minibatch)
            y_test_pred_list.append(y_minibatch_pred)
            y_test_list.append(y_minibatch)
    y_test_preds = torch.cat(y_test_pred_list, 0)
    y_tests = torch.cat(y_test_list, 0)

    predict = pd.DataFrame(y_scaler.inverse_transform(np.array(y_test_preds)), columns=['LSTM 예측'])
    original = pd.DataFrame(y_scaler.inverse_transform(np.array(y_tests)), columns=['실제값'])
    predict['Time Index'] = range(len(predict))  # 시간축 추가
    original['Time Index'] = range(len(original))

    RMSE = mean_squared_error(original['실제값'], predict['LSTM 예측']) ** 0.5
    print(f'RMSE for {symbol}: {RMSE}')

    MDA = calculate_mda(original['실제값'].values, predict['LSTM 예측'].values)
    print(f'MDA for {symbol}: {MDA:.2f}')

    result = pd.merge(original, predict, on='Time Index')
    result.to_csv(f"{symbol}_lstm_predictions.csv", index=False)
    print(f"Predictions saved to {symbol}_lstm_predictions.csv")


symbols = ["AAPL", "MSFT", "NVDA"]  # 예시 종목 리스트
for symbol in symbols:
    try:
        print(f"Predicting for {symbol}...")
        predict(symbol)
    except Exception as e:
        print(f"Error predicting for {symbol}: {e}")