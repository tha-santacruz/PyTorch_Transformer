import os
import linecache

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tokenizer import get_tokenizer

def _count_generator(reader):
    """from https://pynative.com/python-count-number-of-lines-in-file/"""
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

class OpusTranslationDataset():
    def __init__(self, dataset_name, language_source, language_target, vocab_size=30000, sequence_length=256, val_examples=10000, test_examples=10000):
        self.dataset_name = dataset_name
        self.text_file_source = [fname for fname in os.listdir(f"data/{self.dataset_name}/") if fname.endswith(language_source)][0]
        self.text_file_target = [fname for fname in os.listdir(f"data/{self.dataset_name}/") if fname.endswith(language_target)][0]
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.tokenizer = get_tokenizer(
            tokenizer_path=f"tokenizers/{self.dataset_name}_{self.vocab_size}",
            reference_corpora=[f"data/{self.dataset_name}/{self.text_file_source}", f"data/{self.dataset_name}/{self.text_file_target}"],
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length
        )
        with open(f"data/{self.dataset_name}/{self.text_file_source}", "rb") as fp:
            generator = _count_generator(fp.raw.read)
            self.num_examples = sum(buffer.count(b'\n') for buffer in generator) + 1
        with open(f"data/{self.dataset_name}/{self.text_file_target}", "rb") as fp:
            generator = _count_generator(fp.raw.read)
            if self.num_examples != sum(buffer.count(b'\n') for buffer in generator) + 1:
                raise Exception(f"Dataset error : source and target files have unmatching number of entries")
        
        self.indices = {}
        self.indices["all"] = [i for i in range(self.num_examples)]
        self.indices["set"] = self.indices["all"]
        self.indices["test"] = self.indices["all"][:test_examples]
        self.indices["val"] = self.indices["all"][test_examples:test_examples+val_examples]
        self.indices["train"] = self.indices["all"][test_examples+val_examples::]

    def __len__(self):
        return len(self.indices["set"])
    
    def use_set(self, set_name):
        self.indices["set"] = self.indices[set_name]
    
    def __getitem__(self, idx):
        line_num = self.indices["set"][idx]
        
        line_source = linecache.getline(f"data/{self.dataset_name}/{self.text_file_source}", line_num)[:-2] # to remove space and \n at the end
        line_target = linecache.getline(f"data/{self.dataset_name}/{self.text_file_target}", line_num)[:-2]

        tokenized_source = self.tokenizer.encode(line_source)
        tokenized_target = self.tokenizer.encode(line_target)
        source_ids = torch.tensor(tokenized_source.ids)
        source_mask = torch.tensor(tokenized_source.attention_mask)
        target_ids = torch.tensor(tokenized_target.ids)
        target_mask = torch.tensor(tokenized_target.attention_mask)
        return source_ids, source_mask, target_ids, target_mask

if __name__ == "__main__":

    dataset = OpusTranslationDataset(
        dataset_name="WikiMatrix",
        language_source="fr",
        language_target="it"
    )
    dataset = OpusTranslationDataset(
        dataset_name="SeqSort",
        language_source="u",
        language_target="s",
        vocab_size=1000,
        sequence_length=24,
        val_examples=100,
        test_examples=100
    )
    dataset.use_set("train")
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    print(dataset.tokenizer.decode(batch[0][0].tolist()))
    print(dataset.tokenizer.decode(batch[2][0].tolist()))