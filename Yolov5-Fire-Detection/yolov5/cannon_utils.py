import math

def distance_to_fire(x):
    x = 90 - x
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
    return int(angle/3)

def get_angle(steps):
    return int(steps*3)

def get_steps_final_angle(final_angle):
    if (final_angle <= 45):
        return (get_steps(final_angle), 'w')
    else:
        return (get_steps(final_angle - 45), 's')

def get_angle_from_distance(x):
    angle = int(-600 + 10*x - 0.05*x**2 + 0.0000824*x**3)
    if (angle >= 0 and angle <= 75):
        return angle
    if (angle > 75):
        return 75
    return -1