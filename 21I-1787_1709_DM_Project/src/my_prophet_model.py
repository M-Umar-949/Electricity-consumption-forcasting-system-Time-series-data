# from prophet import Prophet
# import pandas as pd
# import pickle

# # Load your DataFrame
# df = pd.read_csv('data/AEP_hourly.csv')

# # # Convert 'Datetime' column to datetime type
# # df['Datetime'] = pd.to_datetime(df['Datetime'])

# # # Set 'Datetime' column as the index
# # df.set_index('Datetime', inplace=True)

# # # Resample to daily data
# # df_daily = df.resample('D').mean().reset_index()

# # # Rename columns to fit Prophet's/home/umar/Desktop/DM_Project/models/prophet_model.pkl requirements
# # df_daily = df_daily.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})

# # # Initialize Prophet model
# # model = Prophet()

# # # Fit the model
# # model.fit(df_daily)

# # # Forecast future values
# # future = model.make_future_dataframe(periods=365)  # Example: forecast next 365 days
# # forecast = model.predict(future)

# # # Extract only the future forecast values
# # forecast_future = forecast[forecast['ds'] > df_daily['ds'].max()]

# # print(forecast_future)


# # # Save the model and forecast to a pickle file
# # with open('prophet_model.pkl', 'wb') as f:
# #     pickle.dump((model, forecast), f)





# # laoding model file and testing


# with open('/home/moz/Desktop/MOZ/models/prophet_model.pkl', 'rb') as f:
#     model, forecast = pickle.load(f)

# # Forecast future values for the next 12 days
# future_12_days = model.make_future_dataframe(periods=12)
# forecast_12_days = model.predict(future_12_days)

# # Extract the 'ds' and 'yhat' values for the next 12 days
# forecast_next_12_days = forecast_12_days[['ds', 'yhat']][-12:]

# #print(forecast_next_12_days)
# print("prophet")





# # In prophet.py
# from prophet import Prophet
# import pandas as pd
# import pickle

# def run_forecast(hours):
#     # Load model
#     with open('/home/moz/Desktop/MOZ/models/prophet_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     future = model.make_future_dataframe(periods=hours, freq='H')
#     forecast = model.predict(future)

#     forecast = forecast[['ds', 'yhat']].tail(hours)
#     forecast.rename(columns={'ds': 'Timestamp', 'yhat': 'Forecast'}, inplace=True)
#     return forecast

# from prophet import Prophet
# import pandas as pd
# import pickle

# def load_prophet_model(model_path):
#     # Load the model from a pickle file
#     with open(model_path, 'rb') as f:
#         model, _ = pickle.load(f)  # Assuming the model is the first item in the tuple
#     return model

# def run_forecast(hours):
#     model_path = '/home/moz/Desktop/MOZ/models/prophet_model.pkl'
#     model = load_prophet_model(model_path)

#     # Generate future dates to predict
#     future = model.make_future_dataframe(periods=hours, freq='h')

#     # Make predictions
#     forecast = model.predict(future)

#     # Extract and return relevant forecast data
#     forecast_relevant = forecast[['ds', 'yhat']]  # 'ds' for datetime, 'yhat' for predictions
#     return forecast_relevant

from prophet import Prophet
import pandas as pd
import pickle

def load_prophet_model(model_path):
    # Load the model from a pickle file
    with open(model_path, 'rb') as f:
        model, _ = pickle.load(f)  # Assuming the model is the first item in the tuple
    return model

def run_forecast(hours):
    model_path = '/home/umar/Desktop/DM_Project/models/prophet_model.pkl'
    model = load_prophet_model(model_path)

    # Generate future dates to predict
    future = model.make_future_dataframe(periods=hours, freq='H')

    # Make predictions
    forecast = model.predict(future)

    # Select the relevant forecast data
    forecast_relevant = forecast[['ds', 'yhat']].tail(hours)  # Selects only the future forecast data
    forecast_relevant['Timestamp'] = forecast_relevant['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Formats the datetime
    forecast_relevant.rename(columns={'yhat': 'Forecast'}, inplace=True)

    # Return the DataFrame with formatted 'Timestamp' and 'Forecast'
    return forecast_relevant[['Timestamp', 'Forecast']]
