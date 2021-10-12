from dataclasses import dataclass


@dataclass
class BasicConfigData:
    threshold: float = 59.0
    period: int = 20

    def update_values(self, values):
        self.threshold = values["threshold"]
        self.period = values["period"]
