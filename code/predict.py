import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.stats import reciprocal
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load


# Load data and scaler
df = pd.read_csv('Franklin LTV Model/code/franklin_ltv_forecast_data.csv')
scaler = load('scaler.joblib')

# Remove rows null/missing values
df = df.dropna()

# Select predictor and target features
X = df[['year', 'month', 'sub_status', 'previous_month_revenue', 'previous_month_orders', 
        'cumulative_revenue_until_last_month', 'end_of_month_subscriber_count']]

X = scaler.transform(X)


# Load model
loaded_model = tf.keras.models.load_model('ltv_model')

# Make prediction
prediction = loaded_model.predict()

print(prediction)