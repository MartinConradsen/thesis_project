#include <Servo.h>
#define tiltDirPin 2
#define tiltStepPin 3
#define turnDirPin 4
#define turnStepPin 5
#define stepsPerRevolution 200
#define DO A5
#define tiltTurnOffPin 12

Servo fireServo;
Servo holdServo;
int pos = 0;

void setup() {
  pinMode(tiltDirPin, OUTPUT);
  pinMode(tiltStepPin, OUTPUT);
  pinMode(turnDirPin, OUTPUT);
  pinMode(turnStepPin, OUTPUT);

  // Make sure tilt motor doesn't get constant power (for overheating reasons)
  pinMode(tiltTurnOffPin, OUTPUT);
  digitalWrite(tiltTurnOffPin, LOW);

  // Microphone
  pinMode(DO, INPUT); 

  // Avoid firing during setup
  fireServo.attach(11);
  for (pos = 80; pos <= 130; pos += 1) {
    fireServo.write(pos);
    delay(5);
  }

  // Holding rod init. angle
  holdServo.attach(13);
  holdServo.write(30);

  Serial.begin(9600);
  Serial.println();
}

void loop() {
  //int output = analogRead(DO);
  //Serial.println(output);
  while (Serial.available() > 0) {
    char inByte = Serial.read();
    // Next two lines: to get the int corresponding to the entered char
    //int number = inByte;
    //Serial.println(number);
    if (inByte == 119) { // w
      tiltOneStepForward();
    } else if (inByte == 115) { // s
      tiltOneStepBackwards();
    } else if (inByte == 97) { // a
      turnOneStepCounterClockwise();
    } else if (inByte == 100) { // d
      turnOneStepClockwise();
    } else if (inByte == 102) { // f
      fire();
    } else if (inByte == 103) { // g
      resetHoldingRod();
    } else if (inByte == 104) { // h
      turnOnTiltMotor();  
    } else if (inByte == 106) { // j
      turnOffTiltMotor();
    } else if (inByte == 109) { // m
      holdProjectile();
    }
  }
}

void tiltOneStepForward() {
  digitalWrite(tiltDirPin, LOW);
  for (int i = 0; i < 20; i++) {
    digitalWrite(tiltStepPin, HIGH);
    delayMicroseconds(1000);
    digitalWrite(tiltStepPin, LOW);
    delayMicroseconds(1000);
  }
}

void tiltOneStepBackwards() {
  digitalWrite(tiltDirPin, HIGH);
  for (int i = 0; i < 20; i++) {
    digitalWrite(tiltStepPin, HIGH);
    delayMicroseconds(1000);
    digitalWrite(tiltStepPin, LOW);
    delayMicroseconds(1000);
  }
}

void turnOneStepClockwise() {
  digitalWrite(turnDirPin, LOW);
  for (int i = 0; i < 20; i++) {
    digitalWrite(turnStepPin, HIGH);
    delayMicroseconds(1000);
    digitalWrite(turnStepPin, LOW);
    delayMicroseconds(1000);
  }
}

void turnOneStepCounterClockwise() {
  digitalWrite(turnDirPin, HIGH);
    for (int i = 0; i < 20; i++) {
      digitalWrite(turnStepPin, HIGH);
      delayMicroseconds(1000);
      digitalWrite(turnStepPin, LOW);
      delayMicroseconds(1000);
    }
}

void fire() {
  releaseProjectile();
  for (pos = 130; pos >= 80; pos -= 1) {
    fireServo.write(pos);
  }
  delay(200);
  for (pos = 80; pos <= 130; pos += 1) {
    fireServo.write(pos);
    delay(1);
  }
  delay(200);  
  resetHoldingRod();
}

void releaseProjectile() {
  holdServo.write(60);
}

void holdProjectile() {
  holdServo.write(5);  
}

void resetHoldingRod() {
  holdServo.write(30);  
}

void turnOffTiltMotor() {
  digitalWrite(tiltTurnOffPin, LOW);  
}

void turnOnTiltMotor() {
  digitalWrite(tiltTurnOffPin, HIGH);  
}