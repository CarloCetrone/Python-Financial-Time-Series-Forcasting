import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------------
# Data Download
# ---------------------
def download_price_history(ticker: str, period: str = "max", interval: str = "1d") -> list:
    """
    Downloads historical price data using yfinance and returns a list.
    period examples: "max", "10y", "1y"
    interval examples: "1d", "1h", "1m" (if available)
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} with period={period} interval={interval}")
    r = df['Close'].copy().to_numpy()
    return r

# ---------------------
# Data Preparation
# ---------------------

def log_price_history(arr):
    """Returns the natural logarithm of each positive value in the given array."""
    return [math.log(x) for x in arr if x > 0]

def make_dataset_maxmin(log_prices, observation_window, prediction_window):
    """
    Creates normalized datasets using min-max scaling for training and testing.
    Each input sequence is normalized based on its minimum and maximum values.
    """
    input_arr = [log_prices[i:i+observation_window] for i in range(len(log_prices)-observation_window-prediction_window)]
    output_arr = [log_prices[i-1+observation_window:i-1+observation_window+prediction_window] for i in range(len(log_prices)-observation_window-prediction_window)]

    inp = []
    out = []

    for seq in input_arr:
        m = min(seq)
        M = max(seq)
        inp.append([(x - m) / (M - m) for x in seq])

    for seq in output_arr:
        out.append([seq[-1] - seq[0]])

    return inp, out

def make_dataset_difference(log_prices, observation_window, prediction_window):
    """
    Creates datasets using price differences for training and testing.
    Each input sequence is expressed as differences from its last value.
    """
    input_arr = [log_prices[i:i+observation_window] for i in range(len(log_prices)-observation_window-prediction_window)]
    output_arr = [log_prices[i-1+observation_window:i-1+observation_window+prediction_window] for i in range(len(log_prices)-observation_window-prediction_window)]

    inp = []
    out = []

    for seq in input_arr:
        inp.append([x - seq[-1] for x in seq])

    for seq in output_arr:
        out.append([seq[-1] - seq[0]])

    return inp, out

# ---------------------
# Model Class
# ---------------------

class FeedForwardNeuralNetwork(nn.Module):
    """
    Defines a feedforward neural network with multiple hidden layers and ReLU activations.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ---------------------
# Model Evaluation
# ---------------------
def plot_histogram(pred, Y, bins):
    """
    Plots the mean output value for prediction intervals divided into bins.
    """
    size = 2 * max(pred)
    d = 2 * size / bins
    arr1 = []
    arr2 = []

    for j in range(0, bins):
        try:
            a = np.mean([Y[i] for i in range(len(Y)) if pred[i] > -size + j * d and pred[i] < -size + (j + 1) * d])
            if a < float('inf'):
                arr1.append(a)
            else:
                arr1.append(0)
        except:
            arr1.append(0)
        arr2.append(-size + j * d + d / 2)

    plt.plot(arr2, arr1)
    plt.xlabel("Prediction")
    plt.ylabel("Mean Output")
    plt.grid()
    plt.show()

def test_trade(pred, Y):
    """
    Simulates a simple trading strategy based on model predictions.
    - If prediction > mean(pred): model enters a long trade.
    - Otherwise: model takes the opposite position.
    Plots cumulative returns for model, opposite model, and buy & hold strategies.
    """
    val1 = 1
    val2 = 1
    val3 = 1
    arr1 = []
    arr2 = []
    arr3 = []
    count_trade = 0
    count_all = 0

    for i in range(len(Y)):
        if pred[i] > np.mean(pred):
            val1 = val1 * math.exp(Y[i])
            arr1.append(val1)
            arr2.append(val2)
            count_trade += 1
        else:
            val2 = val2 * math.exp(Y[i])
            arr1.append(val1)
            arr2.append(val2)

        count_all += 1
        val3 = val3 * math.exp(Y[i])
        arr3.append(val3)

    print('mean pred:', np.mean(pred))
    print("trades:", count_trade)
    print("all:", count_all)
    print(val1 - 1, val2 - 1, val3 - 1)
    
    plt.plot(arr1, label="Model Strategy")
    plt.plot(arr2, label="Opposite Model Strategy")
    plt.plot(arr3, label="Buy & Hold Strategy")
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
