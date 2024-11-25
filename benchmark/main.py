from benchmark.runs.run import Run


class Benchmark:
    def __init__(self, runs: [Run]):
        self.runs = runs


    def run(self):
        # Preform the tests
        for run in self.runs:
            run.start()
