# 📈 FPT Stock Price Forecasting with DLinear / NLinear

This project implements a **multivariate time-series forecasting pipeline** for FPT stock prices using deep learning models inspired by modern Time Series Forecasting (TSF) research.

The model predicts future stock prices using historical windows of multiple features such as **open, high, low, close, volume**, and **log-transformed close price**.

---

## 🚀 Features

- Multivariate time-series input (OHLCV + log price)
- Sliding window forecasting (14 days → 3 days)
- Iterative long-horizon prediction (up to 100 days)
- Feature normalization using `StandardScaler`
- DLinear / NLinear models
- PyTorch-based training
- GPU support (CUDA)
- Visualization of historical + predicted prices

---

## 🧠 Models

This project supports the following models:

### 1️⃣ DLinear
A decomposition-based linear model that learns **trend** and **seasonal** components separately.

### 2️⃣ NLinear
A normalized linear model that improves stability by learning relative changes from the last observed value.

> ⚠️ A hybrid DLinear + NLinear ensemble was tested but did **not** outperform these standalone models on this dataset.

---

## 📊 Input & Output

### Input Features

The model uses the following features:
open
high
low
close
volume
close_log


All features are normalized using `StandardScaler`.

---

### Sliding Window Setup

| Parameter          | Value |
|-------------------|-------|
| Input window      | 14 days |
| Output window     | 3 days |
| Forecast horizon  | 100 days |

Each training sample looks like:

Input : 14 days × 6 features
Target : next 3 days of close_log


---

## 🗂 Project Structure

```text
.
├── data/
│ └── FPT_train.csv
├── models/
│ ├── dlinear.py
│ └── nlinear.py
├── dataset.py
├── train.py
├── predict.py
├── visualization.py
└── README.md
```

---

## 🔧 Installation

```bash
pip install numpy pandas matplotlib torch scikit-learn
```
## 🏃‍♂️ How to Run
### 1️⃣ Train the model
```bash
python train.py
```
### 2️⃣ Predict next 100 days
```bash
python predict.py
```
### 3️⃣ Visualize

Predictions will be plotted together with historical prices.

## 📉 Loss Function

The model is trained using:
```text
Mean Squared Error (MSE)
```
Loss is computed on the normalized log-close price.

## 📈 Results

The model produces:

- Stable multi-step forecasts

- Smooth long-term trend predictions

- Reduced noise compared to simple linear regression

⚠️ Note: Stock prices are inherently stochastic. This project is for educational and research purposes only and should not be used for real financial decisions.

## 🧪 Why Not Hybrid?

A hybrid DLinear + NLinear ensemble was tested, but:

- It introduced more parameters

- Was harder to optimize

- Did not improve validation performance

- Produced higher training loss

Therefore, the final version uses single-model architectures only.

## 🔮 Future Improvements

- Add LSTM / Transformer comparison

- Add technical indicators (RSI, MACD, MA)

- Add validation split + early stopping

- Add probabilistic forecasting

- Support multi-stock training

## ⚠️ Disclaimer

This project is for educational purposes only.
It is not financial advice.
