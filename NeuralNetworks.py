import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

import kagglehub

# step 1: Load dataset
path = kagglehub.dataset_download("paultimothymooney/stock-market-data")
print("Path to dataset files:", path)

all_csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
random.shuffle(all_csvs)
selected_files = all_csvs[:100]

n_days = 20
results = []

# step 2: Loop through each file
for file in selected_files:
    try:
        symbol = os.path.splitext(os.path.basename(file))[0]
        print(f"\nProcessing: {symbol}")

        df = pd.read_csv(file, on_bad_lines='skip', nrows=3000)
        df.dropna(inplace=True)

        if len(df) < 100:
            print(f"Skipping {symbol} — not enough rows.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Close-Open'] = df['Close'] - df['Open']
        df.reset_index(drop=True, inplace=True)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

        # Generate sequences
        X, Y, deltas = [], [], []
        for i in range(n_days, len(scaled) - 1):
            X.append(scaled[i - n_days:i])
            Y.append(1 if scaled[i + 1][3] > scaled[i][3] else 0)
            deltas.append(df.loc[i + 1, 'Close-Open'])

        if len(X) < 50:
            print(f"Skipping {symbol} — not enough training samples.")
            continue

        X, Y, deltas = np.array(X), np.array(Y), np.array(deltas)
        X_train, X_test, Y_train, Y_test, deltas_train, deltas_test = train_test_split(
            X, Y, deltas, test_size=0.2, shuffle=False
        )

        # step 3: Build and train neural network
        model = Sequential([
            LSTM(64, input_shape=(n_days, X.shape[2])),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=0)

        # step 4: Evaluate model and simulate trade
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        trade_amount = sum([deltas_test[i] for i in range(len(y_pred)) if y_pred[i] == 1])
        pred_ct = sum(y_pred)

        print(f"{symbol} | Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Trade Gain: {trade_amount:.2f}")
        results.append([symbol, accuracy, recall, precision, trade_amount, pred_ct])

    except Exception as e:
        print(f"Skipping {file} due to error: {repr(e)}")

# step 5: Save results to CSV
results_df = pd.DataFrame(results, columns=[
    'symbol', 'accuracy', 'recall', 'precision', 'trade_amount', 'pred_ct'
])
results_df.to_csv("NN_Scores.csv", index=False)
print("\nSaved results to NN_Scores.csv")
display(results_df)

#save csv
from google.colab import files
files.download("NN_Scores.csv")

# step 6: Visualization

# Filter and sort by trade_amount
filtered_df = results_df[results_df['trade_amount'].abs() > 1e-2].sort_values(by='trade_amount', ascending=False)

# Plot
plt.figure(figsize=(25, 6))
plt.bar(filtered_df['symbol'], filtered_df['trade_amount'], color='blue')
plt.title("Trade Gain/Loss per Stock (Neural Network Predictions)")
plt.xlabel("Stock Symbol")
plt.ylabel("Profit ($)")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()



