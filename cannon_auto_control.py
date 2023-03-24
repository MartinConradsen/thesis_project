import serial.tools.list_ports
from fire_detector import *

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/cu.usbmodem141201')
serialInst.baudrate = 9600

fire_detected = False

def extinguish_fire():
    configure_cv2()
    while (not fire_detected):
        fire_detected = find_fire()
        serialInst.write('aaaa'.encode('utf-8'))
    serialInst.write('f'.encode('utf-8'))

extinguish_fire()