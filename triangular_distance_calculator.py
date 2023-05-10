import math

def triangle_sides(x):
    a = 220
    y = 90
    z = 180 - x - y
    
    # Convert angles to radians
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
    
    # Calculate the length of side c
    c = (a / math.sin(z)) * math.sin(x)
    
    # Return the lengths of all sides
    return c

def get_steps(angle):
    if angle == 45:
        return (17, 30)
    elif angle == 30:
        return (10, 14)
    elif angle == 15:
        return (6, 3)
    else:
        # Use linear interpolation to estimate steps
        if angle < 15 or angle > 45:
            return (0, 0)  # Outside of the interpolation range
        else:
            x1, y1 = 15, 6
            x2, y2 = 45, 17
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            steps_to_angle = slope * angle + intercept
            steps_to_zero = steps_to_angle + 14  # Assumes symmetry around 0 degrees
            return int(steps_to_angle), int(steps_to_zero)

def get_angle(steps):
    if steps <= 1:
        return 0  # Assume at 0 degrees if fewer than 3 steps taken
    elif steps >= 20:
        return 45  # Assume at 45 degrees if more than 61 steps taken
    else:
        x1, y1 = 15, 6
        x2, y2 = 45, 17
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        angle = (steps - intercept) / slope
        return round(angle)

def get_angle_from_distance(x):
    return -600 + 10*x - 0.05*x**2 + 0.0000824*x**3

print(get_steps(20))