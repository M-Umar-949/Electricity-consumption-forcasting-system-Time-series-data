# import pickle
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# import pandas as pd


# # df= pd.read_csv('data/AEP_hourly.csv')

# # model = ExponentialSmoothing(df['AEP_MW'], trend='add', seasonal='add', seasonal_periods=12)

# # # Fit the ETS model
# # fitted_model = model.fit()

# # # Generate forecasts
# # forecast_steps = 12  # Example: forecast next 12 steps
# # forecast = fitted_model.forecast(steps=fosrecast_steps)

# # # Print or plot forecasts as needed
# # print(forecast)
# # # Save the fitted ETS model as a pickle file
# # with open('ets_model.pkl', 'wb') as f:s
# #     pickle.dump(fitted_model, f)


# # Loading model file and testing


# # Load the saved ETS model from the pickle file
# with open('/home/moz/Desktop/MOZ/models/ets_model.pkl', 'rb') as f:
#     ets_model = pickle.load(f)

# # Generate forecasts for the next 12 steps
# forecast_next_12_days = ets_model.forecast(steps=12)

# # Create a DataFrame to hold the forecasted values with appropriate date indices
# # You may need to adjust this based on your specific date handling
# forecast_dates = pd.date_range(start='2018-01-02', periods=12, freq='h')
# forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_next_12_days})

# #print(forecast_df)
# print("ets")



# In ets.py
import pandas as pd
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def run_forecast(hours):
    # Load the saved model
    with open('/home/umar/Desktop/DM_Project/models/ets_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Generate forecasts
    forecast = model.forecast(steps=hours)
    forecast_times = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='h')

    # Format results
    result_df = pd.DataFrame({
        'Timestamp': forecast_times,
        'Forecast': forecast
    })
    return result_df