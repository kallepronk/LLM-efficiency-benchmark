import datetime
from abc import ABC

import torch

from codecarbon import OfflineEmissionsTracker
from tqdm.auto import tqdm

from benchmark.dataset import Dataset
from benchmark.runs.run import Run


class PreEncodedRun(Run, ABC):
    def __init__(self, model: str, dataset: Dataset, passes: int):
        super().__init__(model, dataset, passes)
        self.name: str = "pre encoded"

    def start (self):
        tokens: [torch.Tensor] = []
        for i in tqdm(range(self.passes)):
            tokens.append(self.tokenizer.encode(self.dataset.get_item(i), return_tensors='pt'))

        tracker = OfflineEmissionsTracker(
            log_level="warning",
            tracking_mode="machine",
            allow_multiple_runs=True,
            output_file="codecarbon.csv",
            country_iso_code="NLD",
            experiment_id=f"{self.passes}passes-{self.model_name}-{datetime.UTC}"
        )
        tracker.start()
        for i in range(self.passes):
            self.model.generate(tokens[i], max_length=100, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1, no_repeat_ngram_size=2)
        print(tracker.stop())
        print(tracker.final_emissions_data)
