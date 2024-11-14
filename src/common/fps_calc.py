# pylint: disable=C0301
# pylint: disable=C0114
# pylint: disable=W1514
# pylint: disable=R0903

import time
class FPSCalc:
    """
    A class to calculate and log the average frames per second (FPS).

    This class keeps track of the number of frames processed and logs the average
    FPS at a specified frequency.
    """
    def __init__(self, freq=60):
        self.start_time_stamp = 0
        self.last_time_stamp = 0
        self.frame_count = -1
        self.prev_frame_count = -1
        self.fps = 0
        self.freq = freq

    def update_fps(self) -> None:
        """
        Update the frame count and calculate the average FPS.
        """
        self.frame_count = self.frame_count + 1
        now = time.time()

        if self.start_time_stamp == 0:
            self.start_time_stamp = time.time()
            self.last_time_stamp = self.start_time_stamp
            with open('FPS.log', 'a') as file:
                file.write('BEV demo log:\n')

        if now >= self.last_time_stamp + self.freq:
            fps = f'{(self.frame_count - self.prev_frame_count) / (now - self.last_time_stamp)}'
            self.fps = fps
            with open('FPS.log', 'a') as file:
                file.write(f'{self.frame_count}, {fps}\n')
            print(f'The AVG FPS is {fps}')
            self.last_time_stamp = now
            self.prev_frame_count = self.frame_count

    def get_fps(self):
        return self.fps
