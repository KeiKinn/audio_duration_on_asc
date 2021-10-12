from dataclasses import dataclass


@dataclass
class BasicConfigData:
    threshold: float = 59.0

    def update_values(self, values):
        self.threshold = values["threshold"]
