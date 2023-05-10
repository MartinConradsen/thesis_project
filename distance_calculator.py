import numpy as np
from numpy.polynomial.polynomial import polyfit

# Heights of the cannon
heights = np.array([220, 220, 220, 220, 220, 220, 110, 110, 110, 110, 110, 110])
heights = heights.flatten()

# Angles of the cannon in radians
angles = np.radians(np.array([45, 30, 15, 0, -15, -30, -45, -30, -15, 0, 15, 30]))
angles = angles.flatten()

# Distances of the projectile
distances = np.array([132.53, 159.50, 207.03, 275.47, 281.80, 283.27, 90.63, 116.90, 145.33, 184.30, 226.87, 231.17])

# Fit a polynomial of degree 2
coefs = polyfit((heights, angles), distances, deg=2)
print(coefs)
#distance = 4.87418863e-03 * height^2 - 4.98298172e-01 * height * angle + 1.82687335e+02 * angle^2

def calculate_distance(height, angle):
    a, b, c = coefs
    return a * height ** 2 + b * height * angle + c * angle ** 2


print(calculate_distance(110, np.radians(15)))