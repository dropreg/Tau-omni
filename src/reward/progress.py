import sys
import threading


class FennecProgressBar:
    def __init__(self, total: int, width: int = 40, stream=None):
        self.total = max(total, 1)
        self.width = max(width, 3)
        self.stream = stream or sys.stderr
        self.fennec = "🦊"
        self.bean = "."
        self.eaten = "="
        self.current = 0
        self.done_count = 0
        self.skip_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def render(self) -> None:
        percent = int(self.current / self.total * 100)
        filled = int(self.current / self.total * self.width)
        fennec_pos = self.width - 1 if self.current >= self.total else min(filled, self.width - 1)

        bar_chars = [self.bean] * self.width
        for i in range(fennec_pos):
            bar_chars[i] = self.eaten
        bar_chars[fennec_pos] = self.fennec
        bar = "".join(bar_chars)
        line = (
            f"[{bar}] {percent:3d}% "
            f"done={self.done_count} skip={self.skip_count} error={self.error_count} "
            f"total={self.total}"
        )
        self.stream.write("\r\033[2K")
        self.stream.write(line)
        self.stream.flush()

    def update(self, *, skipped: bool = False, error: bool = False) -> None:
        with self.lock:
            self.current = min(self.current + 1, self.total)
            if skipped:
                self.skip_count += 1
            else:
                self.done_count += 1
            if error:
                self.error_count += 1
            self.render()

    def finish(self) -> None:
        with self.lock:
            self.current = self.total
            self.render()
            self.stream.write("\n")
            self.stream.flush()
