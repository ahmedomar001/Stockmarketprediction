import os
import glob
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# import dataset from kaggle 
import kagglehub 
path = kagglehub.dataset_download("paultimothymooney/stock-market-data")
print("Path to dataset files:", path)

# read all of the stocks CSVs 
# limit to a few files to avoid memory crash
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[:5]
df_list = [pd.read_csv(f, on_bad_lines='skip', nrows=3000) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)


# clean and sort data
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)  
df.dropna(subset=['Date'], inplace=True)  # drop rows where date parsing failed
df.sort_values('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Only keep numeric features


# normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# use the past ten days to predict stockmarket
X, Y = [], []
n_days = 10 

for i in range(n_days, len(scaled_data)-1):
    X.append(scaled_data[i-n_days:i])
    Y.append(1 if scaled_data[i+1][3] > scaled_data[i][3] else 0)

X, Y = np.array(X), np.array(Y)

# split and train the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

model = Sequential([
    LSTM(64, input_shape=(n_days, X.shape[2])),
    Dense(1, activation='sigmoid')  # for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2)

# evauluate tesr model 
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.2f}")
