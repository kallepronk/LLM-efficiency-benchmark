from abc import ABC
from codecarbon import OfflineEmissionsTracker

from benchmark.runs.run import Run


class PassesRun(Run, ABC):
    def start(self):
        pass
