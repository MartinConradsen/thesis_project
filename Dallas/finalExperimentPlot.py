import matplotlib.pyplot as plt
import numpy as np

# Define the data
data = [23, 27, 23, 13, 16, 12, 15, 20, 17, 18]

# Compute the average
avg = np.mean(data)

# Generate the x values
x_values = range(1, len(data) + 1)

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot the data with dots at each data point
ax.plot(x_values, data, marker='o', label='Data')

# Add the average line
ax.axhline(avg, color='r', linestyle='--', label=f'Average: {avg:.2f}')

# Set the y-axis limits
ax.set_ylim([0, 30])


# Add title and labels
ax.set_title('Distance from target for 10 shots')
ax.set_xlabel('Shot')
ax.set_ylabel('Distance from target in cm')

# Show the plot
plt.show()
