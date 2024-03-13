from datasets import load_dataset
from transformers import BertTokenizer
from scipy.stats import entropy
import numpy as np


class UscDatasetBuilder:

    def __init__(self, filepath) -> None:
        self._ds = load_dataset("csv", data_files=filepath, split='train')
        self._cased = True

    # protected members

    # public members
    def add_tokens(self, cased=True) -> None:
        if self._cased:
            self._tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-cased"
            )
        else:
            self._tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased"
            )

        self._ds = self._ds.map(
            lambda row: self._tokenizer(row["text"], return_tensors="np"), 
            batched=True,
            num_proc=12
        )

    def add_shannon_entropy(self):
        self._ds = self._ds.map(lambda x: {'entropy': entropy(x['input_ids'])})
        print(self._ds['entropy'])

    def get_num_rows(self):
        return self._ds.num_rows

ds = UscDatasetBuilder("output/usc26_sections.csv")
ds.add_tokens()
ds.add_shannon_entropy()
