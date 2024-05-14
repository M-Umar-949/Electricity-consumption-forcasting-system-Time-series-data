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

# # order = (4, 0, 2)  # Use appropriate values based on analysis

# # model = ARIMA(df['AEP_MW'], order=order)

# # fitted_model = model.fit()

# # with open('arima_model.pkl', 'wb') as f:
    

# #     pickle.dump(fitted_model, f)
    
    
    
# # Loading and getting predictions 

# with open('/home/moz/Desktop/MOZ/models/arima_model.pkl', 'rb') as f:
#     arima_model = pickle.load(f)

# # Generate predictions for the next 12 days
# forecast_next_12_days = arima_model.forecast(steps=12)

# # Create a DataFrame to hold the forecasted values with appropriate date indices
# # You may need to adjust this based on your specific date handling
# forecast_dates = pd.date_range(start='2018-01-02', periods=12, freq='h')
# forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_next_12_days})

# #print(forecast_df)
# print("arima")


import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

def run_forecast(hours):
    # Load the model
    with open('/home/umar/Desktop/DM_Project/models/arima_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Generate predictions for the next specified number of hours
    forecast = model.forecast(steps=hours)
    forecast_times = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='h')

    # Format results
    result_df = pd.DataFrame({
        'Timestamp': forecast_times,
        'Forecast': forecast
    })
    return result_df
