from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the model and preprocessing tools
with open('fraud_model.pkl', 'rb') as f:
    loaded = pickle.load(f)
    model = loaded['model']
    scaler = loaded['scaler']

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

@app.route('/')
def home():
    return render_template('one.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    trans_date_trans_time = pd.to_datetime(request.form['trans_date_trans_time'])
    cc_num = request.form['cc_num']
    city_pop = float(request.form['city_pop'])
    merch_lat = float(request.form['merch_lat'])
    merch_long = float(request.form['merch_long'])
    merchant = request.form['merchant']
    category = request.form['category']
    amt = float(request.form['amt'])
    gender = request.form['gender']
    dob = pd.to_datetime(request.form['dob'])
    street = request.form['street']
    city = request.form['city']
    state = request.form['state']
    job = request.form['job']
    lat = float(request.form['lat'])
    long = float(request.form['long'])

    # Feature engineering
    data4 = pd.DataFrame({
        'trans_date_trans_time': [trans_date_trans_time],
        'dob': [dob]
    })
    data4['transaction_month'] = data4['trans_date_trans_time'].dt.month
    data4['transaction_day'] = data4['trans_date_trans_time'].dt.day
    data4['transaction_hour'] = data4['trans_date_trans_time'].dt.hour
    data4['transaction_minute'] = data4['trans_date_trans_time'].dt.minute
    data4['transaction_day_of_week'] = data4['trans_date_trans_time'].dt.dayofweek
    data4['transaction_time_of_day'] = data4['transaction_hour'].apply(get_time_of_day)

    data1 = pd.DataFrame({
        'lat': [lat],
        'long': [long],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long]
    })
    data1['distance'] = data1.apply(lambda row: haversine_distance(
        row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
    age = (trans_date_trans_time - dob).days / 365.25

    # Create DataFrame for new data
    new_data = pd.DataFrame({
        'cc_num': [cc_num],
        'amt': [amt],
        'city_pop': [city_pop],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long],
        'transaction_month': [data4['transaction_month'][0]],
        'transaction_day': [data4['transaction_day'][0]],
        'transaction_hour': [data4['transaction_hour'][0]],
        'transaction_minute': [data4['transaction_minute'][0]],
        'transaction_day_of_week': [data4['transaction_day_of_week'][0]],
        'distance': [data1['distance'][0]],
        'age': [age]
    })

    data2 = pd.DataFrame({
        'merchant': [merchant],
        'category': [category],
        'street': [street],
        'city': [city],
        'state': [state],
        'job': [job]
    })


    # Frequency encoding
    categorical_col=['merchant', 'category', 'street', 'city', 'state', 'job']
    for col in categorical_col:
        frequency_encoding = data2[col].value_counts()
        data2[col] = data2[col].map(frequency_encoding)
   
    data3 = pd.DataFrame({
        'gender': [gender],
        'transaction_time_of_day': [data4['transaction_time_of_day'][0]]
    })

    data3 = pd.get_dummies(data3).astype(int)
    expected_columns = [
        "gender_F", "gender_M",
        "transaction_time_of_day_Afternoon", "transaction_time_of_day_Evening",
        "transaction_time_of_day_Morning", "transaction_time_of_day_Night"
    ]

    for col in expected_columns:
        if col not in data3.columns:
            data3[col] = 0
    data3 = data3[expected_columns]

    new_data1 = pd.concat([new_data, data3], axis=1)
    new_data2 = pd.concat([new_data1, data2], axis=1)

    print("Prediction data columns:", new_data2.columns)

    # Ensure all columns are present before scaling
    # columns_to_scale = ['amt', 'city_pop', 'merch_lat', 'merch_long', 'age', 'distance', 
    #                     'merchant_freq', 'transaction_month', 'transaction_day', 
    #                     'transaction_hour', 'transaction_day_of_week', 'category_freq', 
    #                     'street_freq', 'city_freq', 'state_freq', 'job_freq']
    

    # # Fill missing columns with zeros
    # for col in columns_to_scale:
    #     if col not in new_data2.columns:
    #         new_data2[col] = 0
    
    # print("Final prediction data columns:", new_data2.columns)

    
    # Scale features
    scaler1 = RobustScaler()
    new_data2 = scaler1.fit_transform(new_data2)

    # Make prediction
    prediction = model.predict(new_data2)
    prediction_result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    # Return result
    return render_template('two.html', prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
    