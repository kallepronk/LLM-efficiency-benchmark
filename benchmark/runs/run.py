import abc

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel, AutoModelForCausalLM
)

from benchmark.dataset import Dataset


class Run:
    __metaclass__ = abc.ABCMeta
    def __init__(self, model: str, dataset: Dataset, passes: int):
        self.model_name = model
        self.dataset = dataset
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model)
        self.passes: int = passes


    @abc.abstractmethod
    def start(self):
        pass

