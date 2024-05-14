import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load data


# # Train ANN model
# for epoch in range(100):
#     optimizer.zero_grad()
#     outputs = model(ann_input_tensor)
#     loss = criterion(outputs, ann_input_tensor)
#     loss.backward()
#     optimizer.step()

# Save trained model
#torch.save(model.state_dict(), 'hybrid_model.pth')

# Load trained model for testing
def run_forecast(hours):
    df = pd.read_csv('data/AEP_hourly.csv', index_col='Datetime', parse_dates=['Datetime'])

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load ARIMA model
    with open('/home/umar/Desktop/DM_Project/models/arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)

    # Generate predictions for the next 12 days
    forecast_next_12_days = arima_model.forecast(steps=hours)

    # Create a DataFrame to hold the forecasted values with appropriate date indices
    forecast_times = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='h')

    # Create input data for ANN model
    ann_input = pd.DataFrame({'ARIMA_Forecast': forecast_next_12_days})

    # Convert data to PyTorch tensors
    ann_input_tensor = torch.tensor(ann_input.values, dtype=torch.float32)

    # Define ANN model
    class ANN(nn.Module):
        def __init__(self):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(1, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize ANN model, loss function, and optimizer
    model = ANN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_test = ANN()
    model_test.load_state_dict(torch.load('/home/umar/Desktop/DM_Project/models/hybrid_model.pth'))
    
    # Test loaded model
    predictions = model_test(ann_input_tensor).flatten()

    # Create forecast DataFrame
    result_df = pd.DataFrame({'Timestamp': forecast_times, 'Forecast': predictions.detach().numpy()})
    return result_df

#print(forecast_df)






# import pandas as pd
# import torch
# from torch import nn
# from datetime import datetime

# class Net(nn.Module):
#     def __init__(self, input_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.sigmoid(self.fc4(x))
#         return x

# def load_model(model_path):
#     model = Net(4)  # Adjust if your input feature count differs
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# def run_forecast(hours):
#     model = load_model('/home/moz/Desktop/MOZ/models/hybrid_model.pth')
#     current_time = datetime.now()
#     future_times = pd.date_range(start=current_time, periods=hours, freq='H')
#     future_data = pd.DataFrame({
#         'Hour': future_times.hour,
#         'Day': future_times.day,
#         'Month': future_times.month,
#         'Year': future_times.year
#     })

#     # Assuming no scaling:
#     future_tensor = torch.tensor(future_data.values, dtype=torch.float32)

#     # Get predictions from the model
#     with torch.no_grad():
#         predictions = model(future_tensor)

#     # Format results
#     result_df = pd.DataFrame({
#         'Timestamp': future_times,
#         'Forecast': predictions.numpy().flatten()
#     })
#     return result_df