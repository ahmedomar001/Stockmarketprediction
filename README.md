# Stock Market Direction Prediction

This project explores the use of machine learning models—including Long Short-Term Memory (LSTM) neural networks, Logistic Regression, and Support Vector Machines (SVM)—to predict short-term stock price movements based on historical trading data. The objective is to evaluate both predictive accuracy and simulated trading performance based on model predictions.

## Project Overview

Financial markets are inherently noisy and nonlinear. Traditional models often fail to capture temporal dependencies in stock price data. This project applies deep learning techniques—specifically LSTM networks—along with classical machine learning models to capture meaningful patterns in historical price data and evaluate their effectiveness in simulating trading outcomes.

## Models Used

- LSTM (Long Short-Term Memory) Neural Network
- Logistic Regression
- Support Vector Machine (SVM)

Each model is trained to classify whether the stock price will increase on the following day based on historical data.

## Dataset

- Source: Kaggle Stock Market Dataset  
  [paultimothymooney/stock-market-data](https://www.kaggle.com/paultimothymooney/stock-market-data)
- Each CSV file represents a single stock and contains:
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

## Methodology

### Data Preprocessing

- Removed rows with missing values
- Converted `Date` to datetime format and sorted data chronologically
- Created binary classification label:
  - `1` if the next day’s closing price is higher than the current day’s
  - `0` otherwise
- Applied MinMaxScaler to normalize features
- Generated sliding windows of `n = 20` days for feature extraction

### Model Training

- LSTM Model:
  - Architecture: LSTM (64 units) → Dense (1, sigmoid)
  - Loss function: Binary cross-entropy
  - Optimizer: Adam
  - Chronological 80/20 train/test split (no shuffling)
  - Trained for 30 epochs with a batch size of 64

- Logistic Regression and SVM:
  - Flattened time-series sequences
  - Applied PCA for dimensionality reduction
  - Used 70/30 chronological split
  - Evaluated based on classification and trading metrics

### Trading Simulation

- A trade is simulated when the model predicts a gain (label = 1)
- Trade gain = `Close - Open` for that day
- Calculated total gain/loss across all trades
- Results recorded for each stock: accuracy, precision, recall, total trade gain, and number of predicted trades

## Results Summary

| Model               | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Logistic Regression | 0.54     | 0.52      | 0.55   |
| SVM                 | 0.56     | 0.57      | 0.59   |
| LSTM                | 0.61     | 0.64      | 0.66   |

- LSTM provided the best balance of classification performance and simulated trading gain
- Output stored in `NN_Scores.csv`

## Visualization

- A bar chart was created using Matplotlib to show simulated profit/loss per stock based on LSTM predictions

## Files

- `stock_predict_lstm.py` — LSTM training and simulation script
- `svm_lr_models.py` — Logistic Regression and SVM implementation
- `NN_Scores.csv` — Evaluation results per stock
- `README.md` — Project documentation



## Team

- Ahmed Omar 
- Rushali Gurung  
- Nicholas Doerfler


