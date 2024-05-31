# Hourly Electricity Consumption Forecasting System

## Project Overview

This project aims to develop a forecasting system for hourly electricity consumption using various models, including neural networks. The system predicts electricity usage for the next 12 hours based on historical data.

## Dataset

The dataset used for this project is the `AEP_hourly.csv`, obtained from Kaggle. It contains hourly electricity consumption data. The dataset was analyzed to confirm its stationarity using visualizations and the Augmented Dickey-Fuller (ADF) test.

## Preprocessing

1. **EDA**: Performed exploratory data analysis to understand the data distribution and patterns.
2. **Datetime Conversion**: Converted the Date column to a datetime format to be compatible with forecasting models.
3. **Database Insertion**: Preprocessed data was inserted into a MySQL database for efficient retrieval.

## Model Training

We trained several models to forecast electricity consumption:

1. **ARIMA**
   - Parameters: p=4, q=2
   - Mean Absolute Percentage Error (MAPE): 2.04%

2. **SARIMA**
   - Parameters: p=4, q=2, seasonality index=12
   - MAPE: 28%

3. **ETS**
   - Seasonality period set to 12
   - MAPE: 29%

4. **PROPHET**
   - Data converted to daily granularity
   - MAPE: 14%

5. **SVR (Support Vector Regression)**
   - Additional features extracted from datetime column and standardized
   - MAPE: Not specified

6. **LSTM (Long Short-Term Memory)**
   - Built using PyTorch with 2-3 layers, trained for 100 epochs
   - MAPE: 6%

7. **ANN (Artificial Neural Network)**
   - Four layers, trained for 100 epochs
   - MAPE: 11%

8. **Hybrid Model**
   - Combined ARIMA output as input to ANN
   - Improved forecast accuracy compared to standalone ARIMA

## GUI Integration

We integrated the trained models with a frontend interface. Users can select the number of forecasts and the model to use. The interface was built using d3.js and chart.js to create interactive visualizations of the dataset and predictions.

## Installation and Setup
**Clone the repository**:
   ```bash
   git clone https://github.com/M-Umar-949/Electricity-consumption-forcasting-system-Time-series-data.git
   cd Electricity-consumption-forcasting-system-Time-series-data
   ```

ps: The weights of the ARIMA and SARIMA are not uploaded because they are large files. I will upload them on google drive later.
