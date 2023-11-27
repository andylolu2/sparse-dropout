from collections import defaultdict, deque


class Metrics:
    def __init__(self, max_len: int | None = None):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def clear(self):
        self.history.clear()

    def asdict(self) -> dict[str, list]:
        return {k: list(v) for k, v in self.history.items()}
