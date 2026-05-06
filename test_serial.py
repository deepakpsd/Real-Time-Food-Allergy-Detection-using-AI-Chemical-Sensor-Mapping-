from serial_reader import SerialReader
import time

reader = SerialReader(port="COM4", baudrate=115200)

if reader.connect():
    print("Reading sensor values...\n")

    for _ in range(10):
        value = reader.read_value()
        print("Sensor Value:", value)
        time.sleep(1)

    reader.close()
else:
    print("Failed to connect to ESP8266")