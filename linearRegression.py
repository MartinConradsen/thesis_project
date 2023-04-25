import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# data = pd.read_csv('data.csv')

# X = data[['distance', 'height']]
# y = data['angle']

data = a = np.genfromtxt('data.csv', delimiter=',')
# Separate the data into input (angle, height) and output (distance) variables
X = data[:, [1, 2]]
y = data[:, 0]
print(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


new_data = np.array([[100, 100], [200, 100]])
predictions = model.predict(new_data)

plt.scatter(y_pred, y_test, color='blue')
print(X_test)
# plot the actual data
plt.scatter(X_test[0], y_test, color='blue')

# plot the predicted values
plt.plot(X_test[0], y_pred, color='red', linewidth=2)

# set the labels and title
plt.xlabel('Distance')
plt.ylabel('Angle')
plt.title('Linear Regression')

# display the plot
plt.show()
