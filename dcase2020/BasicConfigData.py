from dataclasses import dataclass


# Note: this is easy to transplant
@dataclass
class BasicConfigData:
    threshold: float = 59.0
    period: int = 20
    is_data_save: int = 0

    def update_values(self, values):
        self.threshold = values["threshold"]
        self.period = values["period"]
        self.is_data_save = values["is_data_save"]
