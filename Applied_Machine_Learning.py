import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from project_functions import (
    FeedForwardNeuralNetwork, download_price_history, log_price_history, make_dataset_maxmin, make_dataset_difference, plot_histogram, test_trade
)

# Initializing variables
observation_window = 64
prediction_window = 2
ticker = 'aapl'
period = 'max'
interval = '1h'
EPOCHS = 1000
train_ratio = 0.5
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the price time series for the selected ticker, period, and interval.
prices = download_price_history(ticker, period, interval)

# Obtaining the time series containing the natural logarithm of each value in the previous time series.
log_prices = log_price_history(prices)

# Splitting the obtained time series into two sections to be used during the training and evaluation phases of the model.
train_prices = log_prices[:int(train_ratio * len(log_prices))]
test_prices = log_prices[int(train_ratio * len(log_prices)):]

# Creating datasets to be used during both the training and evaluation phases.
x_train, y_train = make_dataset_difference(train_prices, observation_window, prediction_window)
x_test, y_test = make_dataset_difference(test_prices, observation_window, prediction_window)

# Converting lists into tensors (needed to work with PyTorch).
x_train = torch.tensor(x_train).to(device)
y_train = torch.tensor(y_train).to(device)
x_test = torch.tensor(x_test).to(device)
y_test = torch.tensor(y_test).to(device)

# Initializing neural network, loss function, and optimizer.
model = FeedForwardNeuralNetwork(observation_window, 256, 1)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Starting the training process.

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_output = model(x_test)
        test_loss = criterion(test_output, y_test)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {loss.item():.8f}, "
              f"Test Loss: {test_loss.item():.8f}")

# Saving the model if needed.
# torch.save(model.state_dict(), "model.pth")

# Loading the saved model if needed.
# model.load_state_dict(torch.load("model.pth"))

# Obtaining predictions from the training dataset.
pred = model(x_train).to('cpu').detach().numpy()
y_train = y_train.to('cpu').detach().numpy()

# Showing performance obtained over the training dataset.
plt.scatter(pred, y_train, s=1)
plt.grid()
plt.xlabel("Prediction")
plt.ylabel("Output")
plt.show()                      # Scatter plot
plot_histogram(pred, y_train, 50) # Histogram plot
test_trade(pred, y_train)         # Backtest plot

# Obtaining predictions from the testing dataset.
pred = model(x_test).to('cpu').detach().numpy()
y_test = y_test.to('cpu').detach().numpy()

# Showing performance obtained over the testing dataset.
plt.scatter(pred, y_test, s=1)
plt.grid()
plt.xlabel("Prediction")
plt.ylabel("Output")
plt.show()                      # Scatter plot
plot_histogram(pred, y_test, 50) # Histogram plot
test_trade(pred, y_test)         # Backtest plot
