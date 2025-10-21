import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self._is_running = False

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
        self._is_running = True

    def toc(self, average=True):
        if self._is_running:
            self.diff = time.time() - self.start_time
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls
            self._is_running = False
            if average:
                return self.average_time
            else:
                return self.diff
        return 0.0

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.