import logging
import time

logger = logging.getLogger("pratyaksa.edge.nextion")

class NextionAlert:
    def __init__(self, port="/dev/ttyAMA0", baud=9600):
        self.ser = None
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Nextion display not available on {port}: {e}")

    def send(self, level: str, message: str):
        if self.ser is None:
            return
        try:
            cmd = f'tAlert.txt="{message}"'
            self.ser.write(cmd.encode('ascii') + b'\xff\xff\xff')
            if level == "critical":
                self.ser.write(b'page critical\xff\xff\xff')
            elif level == "warning":
                self.ser.write(b'page warning\xff\xff\xff')
        except Exception as e:
            logger.warning(f"Nextion send failed: {e}")
