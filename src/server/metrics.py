from dataclasses import dataclass


@dataclass
class RequestMetrics:
    arrival: float
    first_token: float | None = None
    finish: float | None = None

    def record_token(self, now: float) -> None:
        if self.first_token is None:
            self.first_token = now

    def record_finish(self, now: float) -> None:
        self.finish = now

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def end_to_end(self) -> float:
        return self.finish - self.arrival
