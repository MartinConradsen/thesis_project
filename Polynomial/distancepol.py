import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('datat.csv')
df2 = pd.read_csv('data.csv')

# calculate the mean of each angle value
mean_df = df2.groupby('Angle', as_index=False).mean()


# assign the mean DataFrame to X and y
X = mean_df.drop(['Angle', 'Height'], axis=1).values[0:3]
X2 = mean_df.drop(['Angle', 'Height'], axis=1).values[3:6]


y = mean_df['Angle'].values[0:3]
y2 = mean_df['Angle'].values[3:6]

lin_reg = LinearRegression()  # creat model object
lin_reg.fit(X, y)  # fits the model to the training data

lin_reg2 = LinearRegression()  # creat model object
lin_reg2.fit(X2, y2)  # fits the model to the training data



# prints the coefficients of the linear regression model
print("predicted angle = " +  str(lin_reg.coef_)  + "x(distance) + " + str(lin_reg.intercept_))

#plt.scatter(X, y, color='red')
#plt.scatter(X2, y2, color='red')
#plt.plot(X, lin_reg.predict(X), color='blue')
#plt.plot(X2, lin_reg2.predict(X2), color='blue')
#plt.title('Angle acording to distance (Linear Regression)')
#plt.xlabel('Distance')
#plt.ylabel('Angle')
#plt.show()
#plt.waitforbuttonpress()


poly_regr = PolynomialFeatures(degree=3)  # our polynomial model is of order
X = mean_df.drop(['Angle', 'Height'], axis=1).values
y = mean_df['Angle'].values
X_poly = poly_regr.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print(X)

# prints the coefficients of the polynomial regression model
print("Polynomial coefficients:", lin_reg_2.coef_)
print("Intercept:", lin_reg_2.intercept_)


# choice of 0.1 instead of 0.01 to make the graph smoother
X_grid = np.arange(130, 290, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))  # reshapes the array to be a matrix
plt.scatter(X, y, color='red')  # plots the training set
plt.axis([100, 300, -10, 80])
plt.plot(X_grid, lin_reg_2.predict(poly_regr.fit_transform(X_grid)),
         color='green')  # plots a polynomial regression line
# adds tittle to the plot
plt.title('Distance acording to angle (Polynomial Regression)')
plt.xlabel('Distance')  # adds label to the x-axis
plt.ylabel('Angle')  # adds label to the y-axis
plt.show()  # prints our plot

plt.waitforbuttonpress()
