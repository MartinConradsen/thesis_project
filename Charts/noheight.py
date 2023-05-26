from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Load the data
data = pd.read_csv('data.csv', names=['angle', 'distance', 'height'])
print(data)

# Split the data into training and testing sets
X = data[['distance']]
y_angle = data['angle']
y_distance = data['distance']
X_train, X_test, y_angle_train, y_angle_test, y_distance_train, y_distance_test = train_test_split(
    X, y_angle, y_distance, test_size=0.2, random_state=42)


# Train the linear regression model for angle prediction
model_angle = LinearRegression()
model_angle.fit(X_train, y_angle_train)

# Predict angle values for the test set
y_angle_pred = model_angle.predict(X_test)

# Evaluate the model for angle prediction using mean squared error
mse_angle = mean_squared_error(y_angle_test, y_angle_pred)

# Train the linear regression model for distance prediction
model_distance = LinearRegression()
model_distance.fit(X_train, y_distance_train)

# Predict distance values for the test set
y_distance_pred = model_distance.predict(X_test)

# Evaluate the model for distance prediction using mean squared error
mse_distance = mean_squared_error(y_distance_test, y_distance_pred)

# Print the mean squared error for both models
print("Mean squared error for angle prediction:", mse_angle)
print("Mean squared error for distance prediction:", mse_distance)


# Load the data

# Train the linear regression model for angle prediction
X = data[['distance']]
y_angle = data['angle']
model_angle = LinearRegression()
model_angle.fit(X, y_angle)

# Predict angle value for a new distance
new_distance = 300.0
new_angle_pred = model_angle.predict([[new_distance]])

# Print the predicted angle value for the new distance
print("Predicted angle value for distance",
      new_distance, "is", new_angle_pred[0])

X = data[['distance']]
y_angle = data['angle']
model_angle = LinearRegression()
model_angle.fit(X, y_angle)

# Predict angle values for a range of distances
x_range = np.arange(0, 100, 1)
y_angle_pred = model_angle.predict(x_range.reshape(-1, 1))

X = data[['distance']]
y_angle = data['angle']
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)
model_angle = LinearRegression()
model_angle.fit(X_poly, y_angle)

# Predict angle values for a range of distances
x_range = np.arange(0, 100, 1)
x_range_poly = poly.fit_transform(x_range.reshape(-1, 1))
y_angle_pred = model_angle.predict(x_range_poly)

# Plot the predicted function for angle as a function of distance
plt.plot(x_range, y_angle_pred, color='blue', label='Predicted angle')
plt.scatter(X, y_angle, color='red', label='Original data')
plt.xlabel('Distance')
plt.ylabel('Angle')
plt.legend()
plt.show()
