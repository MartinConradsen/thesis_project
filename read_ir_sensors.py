import serial

def read_sensors():
    with serial.Serial('/dev/ttyS1', 19200, timeout=1) as ser:
        line = ser.readline()
        print(line)