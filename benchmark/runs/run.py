import abc

from codecarbon.output_methods.emissions_data import EmissionsData
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
        #self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        #self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model)
        self.passes: int = passes
        self.emissions_data: None | EmissionsData = None
        self.name: str = ""
        self.can_complete = True


    @abc.abstractmethod
    def start(self):
        pass

    def has_finished(self) -> bool:
        if self.emissions_data is None and self.can_complete:
            return False
        else:
            return True


