import serial.tools.list_ports
from fire_detector import *
import vonage
from read_ir_sensors import *
import time

ports = serial.tools.list_ports.comports()
for port in ports:
    print(str(port))
serialInst = serial.Serial('/dev/cu.usbmodem141201')
serialInst.baudrate = 9600

client = vonage.Client(key="79267e59", secret="Master23")
sms = vonage.Sms(client)

fire_detected = False

def extinguish_fire():
    configure_cv2()
    while (not fire_detected):
        fire_detected = find_fire()
        serialInst.write('aaaaaa'.encode('utf-8'))
    serialInst.write('f'.encode('utf-8'))
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

#extinguish_fire()

while (True):
    read_sensors()
    time.sleep(1)
