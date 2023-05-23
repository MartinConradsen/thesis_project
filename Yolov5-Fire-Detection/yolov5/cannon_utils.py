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
    angle = int(-663.5747976152466 + 1.02149668e+01*x - 5.03382246e-02*x**2 + 8.24565617e-05*x**3)
    if (angle >= 0 and angle <= 75):
        return angle
    if (angle > 75):
        return 75
    return -1