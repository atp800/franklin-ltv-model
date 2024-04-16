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

# Define Keras model
def build_model(n_units=32, activation='relu', optimizer='adam', learning_rate=0.001):
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model = Sequential([
        Dense(n_units, activation=activation, input_shape=(X_train_scaled.shape[1],)),
        Dense(n_units, activation=activation),
        Dense(n_units, activation=activation),
        Dense(1)  # Output layer for regression without activation function
    ])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Load data
df = pd.read_csv('Franklin LTV Model/code/franklin_ltv_forecast_data.csv')

# Remove rows null/missing values
df = df.dropna()

# Select predictor and target features
X = df[['year', 'month', 'sub_status', 'previous_month_revenue', 'previous_month_orders', 
        'cumulative_revenue_until_last_month', 'end_of_month_subscriber_count']]
y = df['avg_spend']

# Convert catagorical variable to numerical
X = pd.get_dummies(X, columns=['sub_status'])

# train/test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save fitted scaler to use for making predictions
dump(scaler, 'scaler.joblib')

# Wrap the Keras model as a scikit-learn compatible estimator
keras_regressor = KerasRegressor(build_fn=build_model)

# Hyperparameter search grid
param_grid = {
    'n_units': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [0.01, 0.001, 0.0001],
}

# Wrap the Keras model as a scikit-learn compatible estimator
keras_regressor = KerasRegressor(build_fn=build_model, epochs=100, batch_size=30, verbose=0)

#Grid search with 3-fold cross-validation
grid_search_cv = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, cv=3, verbose=2)

# Run grid search
history = grid_search_cv.fit(X_train_scaled, y_train, validation_split=0.1,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)])

# Get best parameters and evaluate on test set
print("Best parameters:", grid_search_cv.best_params_)
best_params = grid_search_cv.best_params_
print("Best parameters:", best_params)

# Define the best model using the best_params
best_model = build_model(
    n_units=best_params['n_units'],
    activation=best_params['activation'],
    optimizer=best_params['optimizer'],
    learning_rate=best_params['learning_rate']
)

# Fit the best model using the full training set and save the history
history = best_model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    batch_size=30,
    verbose=1,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
)

# Evaluate the best model on the test set
test_loss = best_model.evaluate(X_test_scaled, y_test)
rmse = np.sqrt(test_loss)
print(f"Test loss: {test_loss}")
print(f"RMSE on test set: {rmse}")

best_model.save('ltv_model')

# Plot the training and validation loss
plt.figure(figsize=(10, 4))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()