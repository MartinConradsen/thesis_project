import serial.tools.list_ports
from pynput import keyboard

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/ttyUSB0')
serialInst.baudrate = 9600

count = 0
strBuilder = ""
counting = False
moreCount = 0

def on_press(key):
    global count, strBuilder, counting, moreCount
    try:
        if (key.char == 'c'):
            counting = not counting
        elif (counting):
            strBuilder += str(key.char)
        elif (key.char == 'o'): # forward
            n = int(strBuilder)
            strBuilder = ""
            for _ in range(n):
                strBuilder += 'w'
            serialInst.write(strBuilder.encode('utf-8'))
            strBuilder = ""
        elif (key.char == 'p'): # backwards
            n = int(strBuilder)
            strBuilder = ""
            for _ in range(n):
                strBuilder += 's'
            serialInst.write(strBuilder.encode('utf-8'))
            strBuilder = ""
        elif (key.char == 'k'): # left
            n = int(strBuilder)
            strBuilder = ""
            for _ in range(n):
                strBuilder += 'a'
            serialInst.write(strBuilder.encode('utf-8'))
            strBuilder = ""
        elif (key.char == 'l'): # right
            n = int(strBuilder)
            strBuilder = ""
            for _ in range(n):
                strBuilder += 'd'
            serialInst.write(strBuilder.encode('utf-8'))
            strBuilder = ""
        elif (key.char == 'a'):
            moreCount += 1
            print(moreCount)
            serialInst.write('a'.encode('utf-8'))
        elif (key.char == 'd'):
            moreCount += 1
            print(moreCount)
            serialInst.write('d'.encode('utf-8'))
        elif (key.char == 'w'):
            serialInst.write('w'.encode('utf-8'))
        elif (key.char == 's'):
            serialInst.write('s'.encode('utf-8'))
        elif (key.char == 'f'):
            serialInst.write('f'.encode('utf-8'))
        elif (key.char == 'g'):
            serialInst.write('g'.encode('utf-8'))
        elif (key.char == 'h'):
            serialInst.write('h'.encode('utf-8'))
        elif (key.char == 'j'):
            serialInst.write('j'.encode('utf-8'))
        elif (key.char == 'm'):
            serialInst.write('m'.encode('utf-8'))
            moreCount = 0
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    global strBuilder
    if strBuilder != "":
        print(strBuilder)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
