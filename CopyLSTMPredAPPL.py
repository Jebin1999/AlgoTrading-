#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:23:25 2023

@author: jebin
"""

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define the start and end dates for the data
start = '2010-01-01'
end = '2023-11-25' 

# Fetch AAPL data using Alpha Vantage
api_key = "     "  # Your Alpha Vantage API key
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Process to match previous format
df = data.sort_index()  # Ensure chronological order
df = df[['4. close']]   # Keep only close price
df.columns = ['Close']  # Rename to match previous code


# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Define the window size for the LSTM model
window_size = 60

# Create the training data
X_train = []
y_train = []
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the training data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Create the testing data
X_test = []
y_test = df['Close'].values[train_size+window_size:]
for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-window_size:i, 0])
X_test = np.array(X_test)

# Reshape the testing data for LSTM input
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_data = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_data)

# Create a DataFrame to store the predicted prices with dates
predicted_df = pd.DataFrame({'Date': df.index[train_size+window_size:], 'Predicted Price': predicted_prices.flatten()})

# Set the dates as the index for the DataFrame
predicted_df.set_index('Date', inplace=True)

# Visualize the actual and predicted prices
plt.plot(df.index[train_size+window_size:], y_test, color='blue', label='Actual Price')
plt.plot(predicted_df.index, predicted_df['Predicted Price'], color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('TESLA  Stock Price Prediction')
plt.xticks(rotation=45)  # Rotate x-axis dates by 45 degrees
plt.legend()
plt.show()

# Print the actual and predicted prices with date and time
for i in range(len(predicted_df)):
    date_str = predicted_df.index[i].strftime('%Y-%m-%d')
    actual_price = y_test[i]
    predicted_price = predicted_df['Predicted Price'].iloc[i]
    print(f"Date: {date_str}, Actual Price: {actual_price}, Predicted Price: {predicted_price}")

results_df = pd.DataFrame({
    'Date': predicted_df.index.strftime('%Y-%m-%d'),
    'Actual Price': y_test,
    'Predicted Price': predicted_df['Predicted Price'].values
})




# Calculate evaluation metrics
mse = mean_squared_error(y_test, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predicted_prices)
r2 = r2_score(y_test, predicted_prices)

# Print metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Store metrics in a dictionary or DataFrame (optional)
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R-squared'],
    'Value': [mse, rmse, mae, r2]
})


