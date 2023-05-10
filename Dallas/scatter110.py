import matplotlib.pyplot as plt
import numpy as np

# Generate random data
green_minus_45 = np.array([104, 100, 103, 100, 78, 88, 82, 93, 96, 80, 98,
                          94, 92, 99, 98, 90, 80, 85, 88, 82, 75, 82, 89, 79, 95, 97, 90, 96, 97, 89])
green_minus_30 = np.array([127, 128, 114, 106, 117, 116, 117, 113, 122, 123, 118, 125, 129,
                          122, 117, 118, 128, 118, 121, 109, 105, 109, 104, 108, 111, 116, 114, 119, 117, 116])
green_minus_15 = np.array([157, 148, 144, 146, 136, 151, 135, 132, 136, 138, 148, 139, 147,
                          146, 141, 147, 143, 135, 145, 153, 150, 151, 148, 149, 150, 156, 149, 149, 138, 153])
green_0 = np.array([190, 184, 173, 184, 180, 181, 184, 192, 183, 175, 181, 178, 188,
                   174, 194, 182, 187, 187, 198, 173, 192, 181, 182, 192, 173, 192, 189, 169, 187, 204])
green_15 = np.array([237, 225, 236, 217, 232, 218, 227, 236, 233, 237, 223, 221, 228,
                    231, 217, 232, 226, 223, 227, 229, 225, 218, 214, 233, 229, 227, 223, 227, 226, 229])
green_30 = np.array([226, 235, 223, 232, 227, 224, 231, 230, 228, 233, 227, 233, 238,
                    227, 241, 234, 227, 232, 230, 234, 229, 228, 231, 233, 229, 239, 237, 236, 233, 228])
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

ax.set_title('Green puck at 110cm height')
ax.grid(True)

# Add a legend to the plot
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
legend.set_title('Launch Angle')

plt.show()
