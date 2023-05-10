import matplotlib.pyplot as plt
import numpy as np

# Generate random data
green_minus_45 = np.array([128, 138, 138, 129, 127, 127, 136, 137, 137, 135, 128, 135, 139,
                          134, 133, 127, 135, 126, 129, 128, 140, 139, 139, 127, 134, 129, 140, 127, 128, 127])
green_minus_30 = np.array([151, 163, 160, 161, 152, 154, 163, 174, 165, 141, 151, 162, 162,
                          157, 164, 163, 166, 179, 148, 164, 168, 170, 154, 157, 147, 155, 156, 161, 154, 163])
green_minus_15 = np.array([207, 199, 208, 219, 210, 202, 208, 208, 201, 213, 205, 203, 211,
                          197, 199, 209, 206, 215, 209, 208, 204, 207, 205, 209, 210, 213, 202, 211, 199, 214])
green_0 = np.array([269, 253, 274, 268, 272, 280, 269, 274, 274, 271, 276, 269, 285,
                   279, 284, 278, 268, 277, 271, 274, 282, 281, 278, 282, 283, 287, 278, 273, 281, 274])
green_15 = np.array([284, 284, 286, 285, 284, 291, 287, 284, 277, 273, 271, 284, 285,
                    283, 283, 279, 277, 282, 283, 280, 288, 290, 282, 278, 280, 273, 276, 284, 286, 275])
green_30 = np.array([279, 276, 284, 278, 279, 285, 280, 286, 285, 281, 284, 290, 278,
                    279, 278, 282, 286, 282, 273, 294, 290, 282, 281, 281, 280, 285, 289, 286, 287, 298])
y1 = np.arange(1, 31)
labels = np.random.randint(0, 3, 30)  # generate random labels (0, 1, or 2)

# Create a scatter plot with different color labels
fig, ax = plt.subplots()
scatter = ax.scatter(green_minus_45, y1, label="-45°")
scatter = ax.scatter(green_minus_30, y1, label="-30°")
scatter = ax.scatter(green_minus_15, y1, label="-15°")
scatter = ax.scatter(green_0, y1, label="0°")
scatter = ax.scatter(green_15, y1, label="15°")
scatter = ax.scatter(green_30, y1, label="30°")

ax.set_xlabel('Distance travelled in cm')
ax.set_ylabel('Shot')

ax.set_title('Green puck at 220cm height')
ax.grid(True)

# Add a legend to the plot
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
legend.set_title('Launch Angle')

plt.show()
