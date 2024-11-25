from datasets import load_dataset, DatasetDict


class Dataset:
    def __init__(self, name: str, split: str, column: str):
        self.name = name
        self.split = split
        self.column = column
        self.content: DatasetDict = load_dataset(self.name, split=self.split)

    def get_item(self, row: int):
        return self.content[self.column][row]
