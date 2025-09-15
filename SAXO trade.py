import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# LSTM Model for Time Series Forecasting
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Function to get UIC for a symbol and asset type
def get_uic(base_url, headers, symbol, asset_type):
    url = f"{base_url}/ref/v1/instruments/?Keywords={symbol}&AssetTypes={asset_type}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(f"Error fetching instrument: {resp.text}")
    data = resp.json()
    if 'Data' in data and data['Data']:
        return data['Data'][0]['Uic']
    else:
        raise ValueError(f"Instrument not found for {symbol} {asset_type}")

# Function to fetch historical OHLC data
def get_historical_data(base_url, headers, uic, asset_type, horizon, count=1200):
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    url = f"{base_url}/chart/v1/charts"
    params = {
        "Uic": uic,
        "AssetType": asset_type,
        "Horizon": horizon,
        "Count": min(count, 1200),  # Max 1200 per request
        "Mode": "UpTo",
        "Time": current_time
    }
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise ValueError(f"Error fetching data: {resp.text}")
    data = resp.json()
    if 'Data' not in data:
        return pd.DataFrame()
    ohlc = data['Data']
    # For FX, use mid prices; for others, assume direct OHLC
    if 'OpenBid' in ohlc[0]:
        df = pd.DataFrame([{
            'Time': d['Time'],
            'Open': (d['OpenBid'] + d['OpenAsk']) / 2,
            'High': (d['HighBid'] + d['HighAsk']) / 2,
            'Low': (d['LowBid'] + d['LowAsk']) / 2,
            'Close': (d['CloseBid'] + d['CloseAsk']) / 2
        } for d in ohlc])
    else:
        df = pd.DataFrame([{
            'Time': d['Time'],
            'Open': d['Open'],
            'High': d['High'],
            'Low': d['Low'],
            'Close': d['Close']
        } for d in ohlc])
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time')
    return df

# Function to prepare data for LSTM
def prepare_data(df, seq_length=60):
    data = df[['Open', 'High', 'Low', 'Close']].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    x, y = [], []
    for i in range(seq_length, len(data_normalized)):
        x.append(data_normalized[i-seq_length:i])
        y.append(data_normalized[i])
    x, y = np.array(x), np.array(y)
    return torch.from_numpy(x).float(), torch.from_numpy(y).float(), scaler

# Function to train LSTM model
def train_model(model, x_train, y_train, epochs=10, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        for seq, labels in zip(x_train, y_train):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                  torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
    return model

# Function to predict future steps
def predict_future(model, last_seq, scaler, steps):
    predictions = []
    current_seq = last_seq.copy()  # Last sequence
    for _ in range(steps):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                              torch.zeros(1, 1, model.hidden_layer_size))
        with torch.no_grad():
            pred = model(Variable(torch.from_numpy(current_seq).float()))
        predictions.append(scaler.inverse_transform(pred.numpy().reshape(1, -1))[0])
        # Shift sequence
        current_seq = np.append(current_seq[1:], pred.numpy().reshape(1, -1), axis=0)
    return predictions

# Function to send message to Telegram
def send_to_telegram(telegram_token, chat_id, message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    requests.get(url, params=params)

# Main logic
st.title("Trading Bot Predictions")
st.write("Running predictions and sending to Telegram...")

# Load from secrets
saxo_token = st.secrets["saxo_token"]
telegram_token = st.secrets["telegram_token"]
chat_id = st.secrets["chat_id"]
n_days = int(st.secrets["n_days"])
m_hours = int(st.secrets["m_hours"])

assets = [
    ("XAUUSD", "FxSpot"),
    ("USNAS100.I", "CfdOnIndex")
    # Add more as ("symbol", "AssetType"), e.g., ("EURUSD", "FxSpot")
]
base_url = "https://gateway.saxobank.com/sim/openapi"  # Use /live for live account
headers = {"Authorization": f"Bearer {saxo_token}"}

# Get UICs
uics = {}
asset_types = {}
for symbol, asset_type in assets:
    uics[symbol] = get_uic(base_url, headers, symbol, asset_type)
    asset_types[symbol] = asset_type

# Models: per asset, per timeframe (daily/hourly)
models = {}
scalers = {}
seq_length = 60

messages = []
for symbol in uics:
    for timeframe in ['daily', 'hourly']:
        key = f"{symbol}_{timeframe}"
        horizon = 1440 if timeframe == 'daily' else 60
        steps = n_days if timeframe == 'daily' else m_hours

        # Fetch latest data (full 1200 candles each time)
        df = get_historical_data(base_url, headers, uics[symbol], asset_types[symbol], horizon)
        
        if not df.empty and len(df) > seq_length:
            x_train, y_train, scalers[key] = prepare_data(df, seq_length)
            models[key] = LSTM()
            models[key] = train_model(models[key], x_train, y_train)
            
            # Predict
            data_normalized = scalers[key].transform(df[['Open', 'High', 'Low', 'Close']].values)
            last_seq = data_normalized[-seq_length:]
            preds = predict_future(models[key], last_seq, scalers[key], steps)
            msg = f"{symbol} {timeframe} predictions:\n"
            for i, p in enumerate(preds, 1):
                msg += f"{i}: Open={p[0]:.2f}, High={p[1]:.2f}, Low={p[2]:.2f}, Close={p[3]:.2f}\n"
            messages.append(msg)
        else:
            messages.append(f"{symbol} {timeframe}: Insufficient data for predictions.")

# Send combined message
if messages:
    full_msg = "\n\n".join(messages)
    send_to_telegram(telegram_token, chat_id, full_msg)
    st.write("Predictions sent to Telegram.")
else:
    st.write("No predictions generated.")