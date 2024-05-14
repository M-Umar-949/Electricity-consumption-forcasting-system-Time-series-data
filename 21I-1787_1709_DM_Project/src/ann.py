# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Load your dataset
# df = pd.read_csv('data/AEP_hourly.csv')

# # # Convert 'Datetime' column to datetime type
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# # Extract features from datetime
# df['Hour'] = df['Datetime'].dt.hour
# df['Day'] = df['Datetime'].dt.day
# df['Month'] = df['Datetime'].dt.month
# df['Year'] = df['Datetime'].dt.year

# # Separate features (X) and target variable (y)
# X = df[['Hour', 'Day', 'Month', 'Year']].values  # Features
# y = df['AEP_MW'].values  # Target variable

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling
# scaler_X = StandardScaler()
# X_train_scaled = scaler_X.fit_transform(X_train)
# X_test_scaled = scaler_X.transform(X_test)

# scaler_y = StandardScaler()
# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# # # Convert data to PyTorch tensors
# # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# # X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
# # y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# # # Define batch size
# # batch_size = 64

# # # Create DataLoader for training data
# # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # # Define a more complex feedforward neural network with sigmoid activation
# # class Net(nn.Module):
# #     def __init__(self, input_size):
# #         super(Net, self).__init__()
# #         self.fc1 = nn.Linear(input_size, 128)  # Increased layer size
# #         self.fc2 = nn.Linear(128, 64)  # Additional hidden layer
# #         self.fc3 = nn.Linear(64, 32)
# #         self.fc4 = nn.Linear(32, 1)
# #         self.relu = nn.ReLU()
# #         self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

# #     def forward(self, x):
# #         x = self.relu(self.fc1(x))
# #         x = self.relu(self.fc2(x))
# #         x = self.relu(self.fc3(x))
# #         x = self.sigmoid(self.fc4(x))  # Apply sigmoid activation to output layer
# #         return x

# # # Initialize the neural network
# # input_size = X_train_tensor.shape[1]
# # model = Net(input_size)

# # # Define loss function and optimizer
# # criterion = nn.MSELoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Training loop with batch processing
# # epochs = 200
# # for epoch in range(epochs):
# #     running_loss = 0.0
# #     for inputs, targets in train_loader:
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs.squeeze(), targets)
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()

# #     if epoch % 10 == 0:
# #         print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')

# # # Evaluation
# # model.eval()
# # with torch.no_grad():
# #     y_pred_tensor = model(X_test_tensor)
# # y_pred = scaler_y.inverse_transform(y_pred_tensor.numpy().reshape(-1, 1)).flatten()
# # y_test_inverse = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# # # Calculate metrics
# # mse = mean_squared_error(y_test_inverse, y_pred)
# # print(f"Mean Squared Error (MSE): {mse}")


# # model_path = 'ANN.pth'  # Adjust the path as needed
# # torch.save(model.state_dict(), model_path)

# # print(f"Model saved to {model_path}")




# #***************************************************#
# # loading and testing model
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


# # Define your neural network architecture
# class Net(nn.Module):
#     def __init__(self, input_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)  # Increased layer size
#         self.fc2 = nn.Linear(128, 64)  # Additional hidden layer
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.sigmoid(self.fc4(x))  # Apply sigmoid activation to output layer
#         return x

# # Assuming 'model_path' is the path to your saved model
# model_path = '/home/moz/Desktop/MOZ/models/ANN.pth'

# # Define your model instance and load the state dictionary
# input_size = 4  # Adjust based on your input features
# hidden_size1 = 64  # Adjust based on your model architecture
# hidden_size2 = 32  # Adjust based on your model architecture
# model = Net(input_size)
# model.load_state_dict(torch.load(model_path))


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
# print(result_df[['Timestamp', 'Prediction']])
# print("ANN")







# # In ann.py
# import pandas as pd
# import torch
# from torch import nn
# import numpy as np

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(4, 128)  # Assuming 4 input features: Hour, Day, Month, Year
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         return self.fc4(x)

# def simple_scale(data, feature_range=(0, 1)):
#     # Manually normalize data assuming known ranges
#     min_vals = np.array([0, 1, 1, 2000])  # Hypothetical min values for Hour, Day, Month, Year
#     max_vals = np.array([23, 31, 12, 2023])  # Hypothetical max values for Hour, Day, Month, Year
#     data_scaled = (data - min_vals) / (max_vals - min_vals)
#     data_scaled = data_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
#     return data_scaled

# def simple_inverse_scale(data, feature_range=(0, 1)):
#     # Manually denormalize predictions assuming known ranges
#     min_val = 0  # Hypothetical min value for predictions
#     max_val = 20000  # Hypothetical max value for predictions
#     data_unscaled = data * (max_val - min_val) + min_val
#     return data_unscaled

# def run_forecast(hours):
#     model_path = '/home/moz/Desktop/MOZ/models/ANN.pth'
#     model = Net()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     # Create future data for prediction
#     last_known_time = pd.Timestamp.now()
#     future_times = pd.date_range(start=last_known_time, periods=hours, freq='H')
#     future_data = np.array([
#         future_times.hour,
#         future_times.day,
#         future_times.month,
#         future_times.year
#     ]).T

#     # Normalize the data
#     future_normalized = simple_scale(future_data)

#     # Predict using the model
#     future_tensor = torch.tensor(future_normalized, dtype=torch.float32)
#     with torch.no_grad():
#         predictions = model(future_tensor)
#     predictions = predictions.numpy().flatten()

#     # Denormalize the predictions
#     predictions_denormalized = simple_inverse_scale(predictions)

#     # Prepare result DataFrame
#     result_df = pd.DataFrame({
#         'Timestamp': future_times,
#         'Prediction': predictions_denormalized
#     })

#     return result_df


import pandas as pd
import torch
from torch import nn
import numpy as np

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Assuming input features are: Hour, Day, Month, Year
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def run_forecast(hours):
    model_path = '/home/umar/Desktop/DM_Project/models/ANN.pth'
    model = ANN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create future data for prediction based on the current datetime
    last_known_time = pd.Timestamp.now()
    future_times = pd.date_range(start=last_known_time, periods=hours, freq='h')
    future_data = pd.DataFrame({
        'Hour': future_times.hour,
        'Day': future_times.day,
        'Month': future_times.month,
        'Year': future_times.year
    })

    # Convert DataFrame to tensor for prediction
    future_tensor = torch.tensor(future_data.values, dtype=torch.float32)

    # Generate predictions
    with torch.no_grad():
        predictions = model(future_tensor).numpy().flatten()

    # Create a DataFrame to store predictions alongside timestamps
    result_df = pd.DataFrame({
        'Timestamp': future_times,
        'Forecast': predictions
    })

    return result_df