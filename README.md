# Stock Price Prediction with LSTM

A PyTorch LSTM that predicts a stock's next-day closing price from the previous 29 days of closing prices. It uses daily data from Yahoo Finance, with NVIDIA (`NVDA`) as the default ticker.

## How it works

1. **Data:** Downloads daily price history from 2020-01-01 to today using `yfinance`.
2. **Preprocessing:** Standardizes closing prices with `StandardScaler`.
3. **Sequences:** Splits the series into sliding 30-day windows. The first 29 days of each window are the input, and the 30th day is the target.
4. **Train/test split:** Uses the first 80% of windows for training and the last 20% for testing, in time order.
5. **Model:** A 2-layer LSTM (hidden size 32) followed by a linear layer. It's trained for 200 epochs with Adam (learning rate 0.01) and MSE loss.
6. **Evaluation:** Converts predictions back to dollar prices and reports train and test RMSE.
7. **Visualization:** Plots actual vs. predicted prices for the test period, with a panel showing the absolute prediction error and the RMSE.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook Main.ipynb
```

Then run all cells. Training runs on the GPU automatically if CUDA is available, and on the CPU otherwise.

To try another stock, change the ticker in the third cell:

```python
ticker = 'AAPL'
```

## Configuration

| Setting | Default | Where |
|---|---|---|
| Ticker | `NVDA` | `ticker` |
| Start date | `2020-01-01` | `yf.download(...)` |
| Window length | 30 days | `seq_length` |
| Train split | 80% | `train_size` |
| Hidden size / layers | 32 / 2 | `PredictionModel(...)` |
| Epochs | 200 | `num_epochs` |
| Learning rate | 0.01 | `optim.Adam(...)` |

## Limitations

- The scaler is fit on the full dataset before the train/test split, so some information from the test period leaks into training.
- The model uses only closing prices, with no volume or technical indicators.
- The results haven't been compared against a naive baseline (predicting that tomorrow's price equals today's). For daily prices, that baseline is hard to beat.
- The model predicts raw price levels, so it struggles when the test period's prices are far outside the training range.

## Roadmap

- [ ] Fit the scaler on training data only
- [ ] Add a naive baseline and directional accuracy
- [ ] Predict returns instead of prices
- [ ] Add features (OHLCV, moving averages, RSI, MACD)
- [ ] Add a validation set, early stopping, and mini-batch training
- [ ] Forecast multiple days ahead

## Disclaimer

This project is for educational purposes only and isn't financial advice.
