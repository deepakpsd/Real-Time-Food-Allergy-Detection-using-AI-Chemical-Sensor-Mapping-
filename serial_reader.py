import serial
import time


class SerialReader:
    def __init__(self, port="COM4", baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"Connected to {self.port}")
            return True
        except Exception as e:
            print(f"Serial connection error: {e}")
            self.ser = None
            return False

    def read_value(self):
        if not self.ser:
            return None

        try:
            line = self.ser.readline().decode("utf-8").strip()
            if line.isdigit():
                return int(line)
            return None
        except Exception as e:
            print(f"Read error: {e}")
            return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")