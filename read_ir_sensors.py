import serial

def ir_fire_detected():
    with serial.Serial('/dev/ttyACM0', 19200, timeout=1) as ser:
        line = ser.readline()
        if "1" in line:
            return True
        #return False
        return True
