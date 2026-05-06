void setup() {
  Serial.begin(115200);
  delay(1000);
}

void loop() {
  int sensorValue = analogRead(A0);
  Serial.println(sensorValue);
  delay(500);
}