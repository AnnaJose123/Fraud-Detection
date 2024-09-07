import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pickle
from math import radians, sin, cos, sqrt, atan2

encoder = OneHotEncoder()
scaler = RobustScaler()


# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load the dataset
dt = pd.read_csv(r"D:\FraudDetection\fraudTest.csv")

# Drop unnecessary columns
dt = dt.drop(columns=['Unnamed: 0', 'first', 'last', 'trans_num'])
dt['trans_date_trans_time'] = pd.to_datetime(dt['trans_date_trans_time'])
dt['dob'] = pd.to_datetime(dt['dob'])

# Handling outliers
for col in ['amt', 'city_pop']:
    q1 = dt[col].quantile(0.25)
    q3 = dt[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((dt[col] < lower_bound) | (dt[col] > upper_bound))
    dt[col] = dt[col].where(~outliers, np.median(dt[col]))

# Feature engineering
dt['transaction_year'] = dt['trans_date_trans_time'].dt.year
dt['transaction_month'] = dt['trans_date_trans_time'].dt.month
dt['transaction_day'] = dt['trans_date_trans_time'].dt.day
dt['transaction_hour'] = dt['trans_date_trans_time'].dt.hour
dt['transaction_minute'] = dt['trans_date_trans_time'].dt.minute
dt['transaction_day_of_week'] = dt['trans_date_trans_time'].dt.dayofweek

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

dt['transaction_time_of_day'] = dt['transaction_hour'].apply(get_time_of_day)
dt['is_weekend'] = dt['transaction_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
dt['distance'] = dt.apply(lambda row: haversine_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
dt['age'] = ((dt['trans_date_trans_time'] - dt['dob']).dt.days / 365.25)

dt.drop(columns=['trans_date_trans_time', 'dob', 'zip', 'transaction_year', 'unix_time', 'is_weekend', 'lat', 'long'], inplace=True)
dt = pd.get_dummies(dt, columns=['gender','transaction_time_of_day'])
for i in dt[['gender_F', 'gender_M', 'transaction_time_of_day_Afternoon',
       'transaction_time_of_day_Evening', 'transaction_time_of_day_Morning',
       'transaction_time_of_day_Night']]:
       dt[i]=dt[i].astype(int)


    # Frequency encoding
categorical_col=['merchant', 'category', 'street', 'city', 'state', 'job']
for col in categorical_col:
        frequency_encoding = dt[col].value_counts()
        dt[col] = dt[col].map(frequency_encoding)



# Split data
X = dt.drop(columns=['is_fraud'])
y = dt['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Resample with SMOTE
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scaling features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Update scaled features in DataFrames
X_train_resampled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Model training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Save model and preprocessing tools
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump({'model': rf_classifier, 'scaler': scaler}, f)  # Save both model and scaler

print("Model and preprocessing tools saved successfully!")