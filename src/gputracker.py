import GPUtil
from threading import Thread
import time
import msvcrt

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        
# Instantiate monitor with a 10-second delay between updates
monitor = Monitor(10)
while(True):
    if msvcrt.kbhit():
        if ord(msvcrt.getch()) == 27:
            break

# Close monitor
monitor.stop()
