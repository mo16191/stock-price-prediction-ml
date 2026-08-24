# Comparative Analysis of Machine Learning Models for Stock Price Prediction

MSc Dissertation project (Data Science and Artificial Intelligence, Queen Mary University of London).

Compares five machine learning models — Linear Regression, Decision Tree, Random Forest, Gradient Boosting Machine, and LSTM — on historical AAPL stock data, then uses LIME and SHAP to explain what actually drives each model's predictions.

## Key findings
- **Random Forest and Gradient Boosting Machine performed best** (lowest MAE/RMSE, highest R²), outperforming the more complex LSTM model
- **Lag_5 (5-day lagged price) and MA_50 (50-day moving average)** were consistently the most influential features across all models, confirmed independently by both LIME (local) and SHAP (global) interpretation methods
- LSTM underperformed relative to simpler tree-based models, likely due to the limited size of the dataset, suggesting that model complexity alone doesn't guarantee better forecasts on smaller financial datasets

## Approach
1. **Data collection** — historical AAPL price/volume data pulled via the `yfinance` API (Jan 2020 – Jun 2024)
2. **Exploratory data analysis** — trend, volume and correlation analysis to understand the data and check for multicollinearity
3. **Feature engineering** — moving averages (MA_20, MA_50), volatility, and lag features (Lag_1, Lag_5, Lag_10) to capture momentum and trend
4. **Modelling** — trained and compared five models using an 80/20 train-test split
5. **Evaluation** — MAE, RMSE and R² across all models
6. **Interpretability** — LIME (local, instance-level explanations) and SHAP (global feature importance) to understand *why* each model predicted what it did, not just how accurate it was

## Tools
Python, pandas, scikit-learn, TensorFlow/Keras (LSTM), yfinance, LIME, SHAP, matplotlib/seaborn

## Files
- `stock_prediction.ipynb` — full analysis notebook: data collection through modelling and interpretation
- `report.docx` — full written dissertation report covering methodology, results and conclusions in detail
