import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Define LSTM model class
# # class LSTMModel(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
# #         super(LSTMModel, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers
# #         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
# #         self.fc = nn.Linear(hidden_size, output_size)

# #     def forward(self, x):
# #         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
# #         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
# #         out, _ = self.lstm(x, (h0, c0))
# #         out = self.fc(out[:, -1, :])  # Take the last time step's output
# #         return out

# # # Load your dataset and preprocess it
df = pd.read_csv('data/AEP_hourly.csv')

# # # Convert 'Datetime' column to datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract features from datetime
df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['Year'] = df['Datetime'].dt.year
# # Separate features (X) and target variable (y)
X = df[['Hour', 'Day', 'Month', 'Year']].values
y = df['AEP_MW'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# # # Convert data to PyTorch tensors
# # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Adjust dimensions
# # X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)  # Adjust dimensions
# # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)  # Adjust dimensions
# # y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)  # Adjust dimensions

# # # Define hyperparameters
# input_size = X_train.shape[1]
# hidden_size = 64
# # batch_size = 64
# # learning_rate = 0.001
# # epochs = 100

# # # Create DataLoader for training data
# # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # # Initialize the LSTM model
# # model = LSTMModel(input_size, hidden_size)

# # # Define loss function and optimizer
# # criterion = nn.MSELoss()
# # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # # Training loop
# # for epoch in range(epochs):
# #     model.train()
# #     running_loss = 0.0
# #     for inputs, targets in train_loader:
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs, targets)  # No need to squeeze here
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()
# #     if epoch % 10 == 0:
# #         print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')

# # # Evaluation
# # model.eval()
# # with torch.no_grad():
# #     y_pred = model(X_test_tensor)

# # # Reshape y_pred before inverse transform
# # y_pred_unscaled = scaler_y.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

# # mse = mean_squared_error(y_test, y_pred_unscaled)
# # print(f"Mean Squared Error (MSE): {mse}")
# # torch.save(model.state_dict(), '/content/drive/MyDrive/lstm_model.pth')
# #

# # Loading and testing 


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # Take the last time step's output
#         return out

# model_path = '/home/moz/Desktop/MOZ/models/lstm_model.pth'

# # Load the model
# model = LSTMModel(input_size, hidden_size)
# model.load_state_dict(torch.load('/home/moz/Desktop/MOZ/models/lstm_model.pth'))

# # Get the last known data
# last_known_data = df.iloc[-1]
# last_known_hour = last_known_data['Hour']
# last_known_day = last_known_data['Day']
# last_known_month = last_known_data['Month']
# last_known_year = last_known_data['Year']

# # Prepare next 12 timestamps
# next_12_timestamps = []
# for i in range(1, 13):  # Next 12 timestamps
#     next_timestamp = pd.Timestamp(year=last_known_year, month=last_known_month, 
#                                   day=last_known_day, hour=last_known_hour) + pd.Timedelta(hours=i)
#     next_12_timestamps.append(next_timestamp)

# # Extract features for the next 12 timestamps
# next_12_data = pd.DataFrame({'Timestamp': next_12_timestamps})
# next_12_data['Hour'] = next_12_data['Timestamp'].dt.hour
# next_12_data['Day'] = next_12_data['Timestamp'].dt.day
# next_12_data['Month'] = next_12_data['Timestamp'].dt.month
# next_12_data['Year'] = next_12_data['Timestamp'].dt.year

# # Assuming you have X_next_data with the features Hour, Day, Month, Year for the next timestamps
# X_next_data = next_12_data[['Hour', 'Day', 'Month', 'Year']].values
# X_next_scaled = scaler_X.transform(X_next_data)

# # Now you can use X_next_data with your trained model to generate predictions
# # Convert data to PyTorch tensors
# X_next_tensor = torch.tensor(X_next_scaled, dtype=torch.float32).unsqueeze(1)  # Adjust dimensions

# # Ensure the model is in evaluation mode
# model.eval()

# # Generate predictions for the next 12 timestamps
# with torch.no_grad():
#     y_pred_next = model(X_next_tensor)

# # Reshape y_pred_next before inverse transform
# y_pred_next_unscaled = scaler_y.inverse_transform(y_pred_next.detach().numpy().reshape(-1, 1))

# # Print or use y_pred_next_unscaled for further analysis
# pred_df = pd.DataFrame({'Timestamp': next_12_timestamps, 'Prediction': y_pred_next_unscaled.flatten()})

# # Merge with next_12_data DataFrame
# result_df = pd.merge(next_12_data, pred_df, on='Timestamp', how='left')

# # Display the result DataFrame
# #print(result_df[['Timestamp', 'Prediction']])
# print("lstm")




# In lstm.py
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from sklearn.preprocessing import StandardScaler
# import pickle

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         return self.fc(hn.squeeze(0))

# def run_forecast(hours):
#     # Load model and scaler
#     model = LSTMModel(4, 64)  # Modify as per actual architecture
#     model.load_state_dict(torch.load('/home/moz/Desktop/MOZ/models/lstm_model.pth'))
#     model.eval()
#     scaler = pickle.load(open('/path/to/scaler.pkl', 'rb'))

#     # Simulate generating input data
#     # Normally you would fetch or simulate this data
#     timestamps = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='H')
#     data = np.random.rand(hours, 4)  # Dummy data; replace with actual features
#     data_scaled = scaler.transform(data)
#     data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

#     with torch.no_grad():
#         predictions = model(data_tensor).squeeze().numpy()

#     result_df = pd.DataFrame({
#         'Timestamp': timestamps,
#         'Forecast': predictions
#     })
#     return result_df


# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # Get last time step outputs
#         return out

# def run_forecast(hours):
#     model_path = '/home/moz/Desktop/MOZ/models/lstm_model.pth'
#     input_size = 4  # Based on your features: Hour, Day, Month, Year
#     hidden_size = 64
#     num_layers = 2  # Make sure this matches the layers used during training
#     output_size = 1
    
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     model.to('cpu')  # Assuming no GPU is used

#     last_known_time = pd.Timestamp.now()  # Assuming forecasting starts now
#     future_times = pd.date_range(start=last_known_time, periods=hours, freq='H')
#     future_data = np.array([
#         future_times.hour,
#         future_times.day,
#         future_times.month,
#         future_times.year
#     ]).T

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler.fit([[0, 1, 1, 2000], [23, 31, 12, 2023]])
#     future_scaled = scaler.transform(future_data)
    
#     future_tensor = torch.tensor(future_scaled, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
#     with torch.no_grad():
#         predictions = model(future_tensor)

#     predictions = predictions.numpy().flatten()

#     result_df = pd.DataFrame({
#         'Timestamp': future_times,
#         'Prediction': predictions
#     })

#     return result_df

import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_model(model_path, input_size, hidden_size, num_layers):
    model = LSTMModel(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load("/home/umar/Desktop/DM_Project/models/lstm_model.pth"))
    model.eval()
    return model

def run_forecast(hours):
    input_size = 4  # Define according to your trained model features
    hidden_size = 64  # Define according to your trained model configuration
    num_layers = 2  # Define according to your trained model configuration

    model = load_model("/home/umar/Desktop/DM_Project/models/lstm_model.pth", input_size, hidden_size, num_layers)

    # Generating timestamp data for prediction
    last_known_time = pd.Timestamp(datetime.now())
    future_times = pd.date_range(start=last_known_time, periods=hours, freq='h')
    future_data = pd.DataFrame({
        'Hour': future_times.hour,
        'Day': future_times.day,
        'Month': future_times.month,
        'Year': future_times.year
    })
    future_data=scaler_X.transform(future_data)
    # Data preparation for LSTM input
    future_tensor = torch.tensor(future_data, dtype=torch.float32).unsqueeze(1)  # Batch dimension

    # Prediction
    with torch.no_grad():
        
        predictions = model(future_tensor)
        predictions= scaler_y.inverse_transform(predictions)
        predictions=predictions.flatten()
        print(predictions)

    # Prepare result DataFrame
    result_df = pd.DataFrame({
        'Timestamp': future_times,
        'Forecast': predictions
    })
    return result_df

