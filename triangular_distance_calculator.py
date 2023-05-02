import math

def triangle_sides(x):
    """
    Calculate the length of the bottom side in a triangle
    given the length of one side (a) and the angles
    opposite to that side (x, y).
    x = cannon angle
    """
    a = 220
    y = 90
    # Calculate the angle opposite to side c
    z = 180 - x - y
    
    # Convert angles to radians
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
    
    # Calculate the length of side c
    c = a * math.sin(z) / math.sin(x)
    
    # Return the lengths of all sides
    return c