# from re import I
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
# import mysql.connector
# import pymysql
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal import seasonal_decompose
# import pickle

# # df= pd.read_csv('data/AEP_hourly.csv')

# # order = (1, 0, 1) 
# # seasonal_order = (1, 0, 1, 12)  

# # # Create and fit SARIMA model
# # sarima_model = SARIMAX(df['AEP_MW'], order=order, seasonal_order=seasonal_order)
# # fitted_sarima_model = sarima_model.fit()

# # # Forecast
# # forecast_steps = 10  # Example: forecast next 12 steps
# # forecast = fitted_sarima_model.forecast(steps=forecast_steps)

# # forecast

# # with open('sarima_model.pkl', 'wb') as f:
# #     pickle.dump(fitted_sarima_model, f)
    
    

# # Loading and tesing model file

# # # Load the SARIMA model from the pickle file
# with open('/home/moz/Desktop/MOZ/models/sarima_model.pkl', 'rb') as f:
#     sarima_model = pickle.load(f)

# # Generate forecasts with the loaded SARIMA model
# forecast_next_12_days = sarima_model.forecast(steps=12)

# # Create a DataFrame to hold the forecasted values with appropriate date indices
# # You may need to adjust this based on your specific date handling
# forecast_dates = pd.date_range(start='2018-01-02', periods=12, freq='h')
# forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_next_12_days})

# #print(forecast_df)
# print("sarima")


import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_forecast(hours):
    try:
        with open('/home/umar/Desktop/DM_Project/models/sarima_model.pkl', 'rb') as f:
            model = pickle.load(f)

        forecast = model.forecast(steps=hours)
        forecast_times = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='h')

        result_df = pd.DataFrame({
            'Timestamp': forecast_times,
            'Forecast': forecast
        })
        return result_df
    except Exception as e:
        # Log the exception or handle it accordingly
        return f"Error occurred: {str(e)}"