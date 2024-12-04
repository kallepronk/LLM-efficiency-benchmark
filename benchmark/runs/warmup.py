from codecarbon.output_methods.emissions_data import EmissionsData

from benchmark.dataset import Dataset
from benchmark.runs.run import Run
from tqdm.auto import tqdm


class Warmup(Run):
    def __init__(self, model: str, dataset: Dataset, passes: int):
        super().__init__(model, dataset, passes)
        self.name: str = "warmup"
        self.can_complete = False

    def start (self):
        for i in tqdm(range(self.passes)):
            input_ids = self.tokenizer.encode(self.dataset.get_item(i), return_tensors='pt')
            output = self.model.generate(input_ids, max_length=60, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1, no_repeat_ngram_size=2)

            for sequence in output:
                self.tokenizer.decode(sequence, skip_special_tokens=True)
