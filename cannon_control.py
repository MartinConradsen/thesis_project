import serial.tools.list_ports
from pynput import keyboard

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/cu.usbmodem141201')
serialInst.baudrate = 9600

def on_press(key):
    try:
        if (key.char == 'a'):
            serialInst.write('a'.encode('utf-8'))
        elif (key.char == 'd'):
            serialInst.write('d'.encode('utf-8'))
        elif (key.char == 'w'):
            serialInst.write('w'.encode('utf-8'))
        elif (key.char == 's'):
            serialInst.write('s'.encode('utf-8'))
        elif (key.char == 'f'):
            serialInst.write('f'.encode('utf-8'))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()