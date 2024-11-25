import abc

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel
)

from benchmark.dataset import Dataset


class Run:
    __metaclass__ = abc.ABCMeta
    def __init__(self, model: str, dataset: str, passes: int):
        self.model_name = model
        self.dataset_name = dataset
        # self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        # self.model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(model)
        # self.dataset: Dataset
        self.passes: int = passes


    @abc.abstractmethod
    def start(self):
        pass

