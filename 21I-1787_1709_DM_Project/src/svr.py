# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import pickle

# # Load your dataset (replace 'your_dataset.csv' with your data file)
# df= pd.read_csv('data/AEP_hourly.csv')
# # # Convert 'Datetime' column to datetime type
# # df['Datetime'] = pd.to_datetime(df['Datetime'])

# # # Extract features from datetime (e.g., day of week, month, etc.)
# # df['DayOfWeek'] = df['Datetime'].dt.dayofweek
# # df['Month'] = df['Datetime'].dt.month
# # df['DayOfMonth'] = df['Datetime'].dt.day
# # df['Hour'] = df['Datetime'].dt.hour

# # # Separate features (X) and target variable (y)
# # X = df.drop(columns=['Datetime', 'AEP_MW'])  # Features
# # y = df['AEP_MW']  # Target variable

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Feature scaling (use StandardScaler or MinMaxScaler)
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Initialize SVR model with chosen kernel and hyperparameters
# # svr_model = SVR(kernel='linear', C=10, gamma='auto')  # RBF kernel

# # # Train the SVR model
# # svr_model.fit(X_train_scaled, y_train)

# # # Save the trained model to a pickle file
# # with open('svr_model.pkl', 'wb') as f:
# #     pickle.dump(svr_model, f)

# # # Make predictions on the test set
# # y_pred = svr_model.predict(X_test_scaled)

# # print(y_pred)
# # # Evaluate the model
# # mse = mean_squared_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)

# # print(f"Mean Squared Error (MSE): {mse}")
# # print(f"R-squared (R2): {r2}")


# #***********************************************
# # Load the trained model from a pickle file



# with open('/home/moz/Desktop/MOZ/models/svr_model.pkl', 'rb') as f:
#     svr_model = pickle.load(f)

# # Assuming you have a DataFrame with timestamps for the next 12 steps
# next_12_timestamps = pd.date_range(start='2018-01-02', periods=12, freq='h')
# next_12_data = pd.DataFrame({'Timestamp': next_12_timestamps})

# # Extract features from the forecasted dates
# next_12_data['DayOfWeek'] = next_12_data['Timestamp'].dt.dayofweek
# next_12_data['Month'] = next_12_data['Timestamp'].dt.month
# next_12_data['DayOfMonth'] = next_12_data['Timestamp'].dt.day
# next_12_data['Hour'] = next_12_data['Timestamp'].dt.hour

# # Prepare X_next based on the extracted features
# X_next = next_12_data[['DayOfWeek', 'Month', 'DayOfMonth', 'Hour']].values

# # Generate forecasts for the next 12 timestamps using the SVR model
# forecast_next_12 = svr_model.predict(X_next)

# # Create a DataFrame to hold the forecasted values with timestamps
# forecast = pd.DataFrame({
#     'Timestamp': next_12_data['Timestamp'],
#     'Forecast': forecast_next_12
# })

# #print(forecast)
# print("svr")



# # In svr.py
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# import pickle

# def run_forecast(hours):
#     with open('/home/moz/Desktop/MOZ/src/svr.py', 'rb') as f:
#         model = pickle.load(f)
#     scaler = pickle.load(open('/path/to/scaler.pkl', 'rb'))

#     # Simulate feature data for forecasting
#     timestamps = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='H')
#     data = np.random.rand(hours, 4)  # Assume 4 features, replace with real feature engineering logic
#     data_scaled = scaler.transform(data)

#     predictions = model.predict(data_scaled)

#     result_df = pd.DataFrame({
#         'Timestamp': timestamps,
#         'Forecast': predictions
#     })
#     return result_df



import pandas as pd
from sklearn.svm import SVR
import pickle

def run_forecast(hours):
    # Load the model
    try:
        with open('/home/umar/Desktop/DM_Project/models/svr_model.pkl', 'rb') as f:
            svr_model = pickle.load(f)
    except Exception as e:
        return {'error': str(e)}

    # Create DataFrame with the exact structure expected by the model
    # Example: Assuming the model was trained with 'Hour', 'Day', 'Month', 'Year'
    # Let's create dummy values for demonstration. In practice, these should be meaningful values.
    current_time = pd.Timestamp.now()
    future_times = pd.date_range(start=current_time, periods=hours, freq='h')

    data = pd.DataFrame({
        'Hour': future_times.hour,
        'Day': future_times.day,
        'Month': future_times.month,
        'Year': future_times.year,
    })

    # Make predictions
    predictions = svr_model.predict(data)
    print(predictions)
    # Prepare result DataFrame
    result_df = pd.DataFrame({
        'Timestamp': future_times,
        'Forecast': predictions
    })

    return result_df