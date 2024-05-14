import pandas as pd
from flask import Flask, request, render_template, jsonify
import src.ann as ann 
import src.arima as arima
import src.ets as ets
import src.hybrid as hybrid
import src.lstm as lstm
import src.my_prophet_model as prophet
import src.sarima as sarima
import src.svr as svr
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('GUI.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    hours = int(request.form['hours'])
    model_type = request.form['model-selector']
    model_map = {
        'ANN': ann,
        'ARIMA': arima,
        'ETS': ets,
        'HYBRID': hybrid,
        'LSTM': lstm,
        'PROPHET': prophet,
        'SARIMA': sarima,
        'SVR': svr
    }

    if model_type in model_map:
        try:
            result = model_map[model_type].run_forecast(hours)
            # Ensure the result is in a format that can be JSON serialized
            if isinstance(result, pd.DataFrame):

                labels = result['Timestamp'].tolist()
                forecasts = result['Forecast'].tolist()
                return jsonify({'labels': labels, 'forecasts': forecasts})
            elif isinstance(result, list):
                labels = [f"Hour {i+1}" for i in range(hours)]
                forecasts = result
                return jsonify({'labels': labels, 'forecasts': forecasts})
            else:
                # Handle other types if necessary
                return jsonify({'error': 'Unsupported data type returned from model.'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Model not selected'}), 400
    
    
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'GET':
        return render_template('dashboard.html')
    elif request.method == 'POST':
        # Read CSV file and aggregate data
        df = pd.read_csv('static/AEP_hourly.csv')  # Adjust the path as needed
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month

        # Aggregate by month and year
        monthly_data = df.groupby(['Year', 'Month']).mean()['AEP_MW'].reset_index()
        yearly_data = df.groupby(['Year']).mean()['AEP_MW'].reset_index()

        # Convert DataFrame to JSON format
        monthly_json = monthly_data.to_json(orient='records')
        yearly_json = yearly_data.to_json(orient='records')

        # Prepare data to send to frontend
        data_to_send = {
            'monthly': json.loads(monthly_json),
            'yearly': json.loads(yearly_json)
        }

        return jsonify(data_to_send)


if __name__ == '__main__':
    app.run(debug=True)