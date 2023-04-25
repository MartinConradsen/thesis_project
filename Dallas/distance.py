import numpy as np
import matplotlib.pyplot as mp
from scipy.optimize import minimize_scalar

# Define your data as a numpy array with columns for angle, distance, and height
# data = np.array([[0, 169, 110], [0, 187, 110], [0, 204, 110],
# [15, 237, 110], [15, 225, 110], [15, 236, 110], [15, 217, 110]])

data = a = np.genfromtxt('newdata.csv', delimiter=',')
# Separate the data into input (angle, height) and output (distance) variables
X = data[:, [0, 2]]
y = data[:, 1]
print(X)
print(" ")
print(y)

# Define a function that computes the predicted distance for a given height and angle


def predict_distance(height, angle, a, b, c, d, e):
    return a * height**2 + b * height + c * angle**2 + d * angle + e

# Define a function that computes the error between the predicted and actual distances


def error_function(angle, height, X, y):
    a, b, c, d, e = np.polyfit(X[:, 0], y - X[:, 1]**2, 4)
    predicted_distance = predict_distance(height, angle, a, b, c, d, e)
    actual_distance = np.interp(height, X[:, 1], y)
    print(predicted_distance)
    print(actual_distance)
    return (predicted_distance - actual_distance)**2

# Define a function that finds the angle needed to achieve a desired distance at a given height


def find_angle_for_distance(height, desired_distance, X, y):
    def distance_error_function(angle, height, X, y, desired_distance):
        a, b, c, d, e = np.polyfit(X[:, 0], y - X[:, 1]**2, 4)
        predicted_distance = predict_distance(height, angle, a, b, c, d, e)
        print(predicted_distance)
        return (predicted_distance - desired_distance)**2
    result = minimize_scalar(distance_error_function, args=(
        height, X, y, desired_distance), bounds=(0, 90), method='bounded')
    return result.x


# Example usage:
height = 120
desired_distance = 200
angle = find_angle_for_distance(height, desired_distance, X, y)
print(
    f"To achieve a distance of {desired_distance} at a height of {height}, you need an angle of {angle} degrees.")
