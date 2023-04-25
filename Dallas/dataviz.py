from matplotlib import pyplot as plt
import pandas as pd
df = pd.read_csv('data.csv', header=None)
df.rename(columns={0: 'angle', 1: 'distance', 2: 'height'}, inplace=True)

angle = df['angle']
distance = df['distance']
height = df['height']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(angle, distance, height)

# Set the plot title and axis labels
ax.set_title('Projectile Motion')
ax.set_xlabel('Angle')
ax.set_ylabel('Distance')
ax.set_zlabel('Height')

# Show the plot
plt.show()
