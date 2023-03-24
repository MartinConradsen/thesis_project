import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/cu.usbmodem141201')
serialInst.baudrate = 9600

fire_detected = False

def find_fire():
    while (fire_detected == False):
        # Look for fire - if found, set fire_detected to True
        if (fire_detected == False):
            serialInst.write('aaaa'.encode('utf-8'))
            continue
    serialInst.write('f'.encode('utf-8'))