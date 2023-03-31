# thesis_project
DTM thesis project 2023.


client = vonage.Client(key="79267e59", secret="Master23")
sms = vonage.Sms(client)

responseData = sms.send_message(
    {
        "from": "FIREARM",
        "to": "4553669936",
        "text": "A text message",
    }
)

if responseData["messages"][0]["status"] == "0":
    print("Message sent successfully.")
else:
    print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
