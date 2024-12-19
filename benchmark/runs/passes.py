import datetime
from abc import ABC
from codecarbon import OfflineEmissionsTracker
from tqdm.auto import tqdm

from benchmark.dataset import Dataset
from benchmark.runs.run import Run


class PassesRun(Run, ABC):
    def __init__(self, model: str, dataset: Dataset, passes: int):
        super().__init__(model, dataset, passes)
        self.name: str = "passes"

    def start (self):
        tracker = OfflineEmissionsTracker(
            log_level="warning",
            tracking_mode="machine",
            allow_multiple_runs=True,
            output_file="codecarbon.csv",
            country_iso_code="NLD",
            experiment_id=f"{self.passes}passes-{self.model_name}-{datetime.UTC}"
        )
        tracker.start()
        pbar = tqdm(total=self.passes, desc=f"{self.name} - passes: {self.passes}")

        for i in range(self.passes):
            input_ids = self.tokenizer.encode(self.dataset.get_item(i), return_tensors='pt')
            output = self.model.generate(input_ids, max_length=100, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1, no_repeat_ngram_size=2)

            for sequence in output:
                self.tokenizer.decode(sequence, skip_special_tokens=True)
            pbar.update()
        tracker.stop()
        self.emissions_data = tracker.final_emissions_data