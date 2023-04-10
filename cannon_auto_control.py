import serial.tools.list_ports
from fire_detector import *
import vonage
from read_ir_sensors import *
import time
import sys

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/ttyACM0')
serialInst.baudrate = 9600

client = vonage.Client(key="79267e59", secret="Master23")
sms = vonage.Sms(client)

fire_detected = False
count = 0
forward = True

def extinguish_fire():
    global fire_detected, count, forward
    serialInst.write('m'.encode('utf-8'))
    time.sleep(0.2)
    serialInst.write('h'.encode('utf-8'))
    time.sleep(0.2)
    serialInst.write('wwwwwwww'.encode('utf-8'))
    configure_cv2()
    while (not fire_detected):
        if (camera_found_fire()):
            print("Camera found fire.")
            #if (ir_fire_detected()):
            if (True):
                print("IR sensors detected fire.")
                print("Firing projectile.")
                fire_detected = True
                serialInst.write('f'.encode('utf-8'))
                time.sleep(1)
                serialInst.write('j'.encode('utf-8'))
                sys.exit()
                break
        if (count < 20 and forward == True):
            serialInst.write('aaaaaa'.encode('utf-8'))
            count += 1
        else:
            forward = False
            serialInst.write('dddddd'.encode('utf-8'))
            count -= 1
        if (count == -1):
            forward = True
    #send_text_alert()

def send_text_alert():
    responseData = sms.send_message(
        {
            "from": "FIRE ALARM",
            "to": "4553669936",
            "text": "The fire extinguishing cannon has detected fire and attempted to extinguish it.",
        }
    )
    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")

extinguish_fire()
