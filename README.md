# PyTorch Implementation of a Transformer Network
## Work in progress
This repository contains a still ongoing personal project. The code is not optimised for speed or memory usage yet. The code is not commented either.

## Project Description
The goal of the project is to implement a Transformer network from scratch. This is achieved by following the explainations provided in the original paper [1].

The implemented modules are
1. The MLP
2. The Attention Head
3. The Multi-Head Attention block
4. The Masked Attention Head
5. The Masked Multi-Head Attention block
6. The Transformer Encoder Layer
7. The Transformer Decoder Layer
8. The Transformer (Embeddings, Encoder, Decoder)

## Tokenizer
The tokenizer uses Byte-Pair Encoding (BPE) to create a vocabulary based on the contents of reference corpora. After creation, tokenizers are saved so they can be directly loaded again without going trough the whole pairs merging process again.

It uses the following special tokens : 
- ```[UNK]``` for unknown tokens
- ```[PAD]``` for padding tokens
- ```[BOS]``` for beginning-of-sequence tokens
- ```[EOS]``` for end-of-sequence tokens

The usage of the tokeniser class and get_tokenizer function is as follows
```python
from tokenizer import get_tokenizer

# Declare
tokenizer = get_tokenizer(
    tokenizer_path="tokenizer_file_path", 
    reference_corpora=[
        "corpus_file_1", "corpus_file_2", "corpus_file_3"
    ], 
    vocab_size=30000, 
    sequence_length=128)

# Encode sentence
encoding = tokenizer.encode("This is an example sentence with an unknown token 😀")
tokens = encoding.tokens
ids = encoding.ids
attention_mask = encoding.attention_mask

# Decode sentence
decoding = tokenizer.decode(ids, skip_special_tokens=False)
```

## Dataset
The dataset class is meant to process datasets that have the following structure, which is adopted from [OPUS Corpora](https://opus.nlpl.eu)

- The data is made of two main files
- One file contains the source text sequences and the other one contains the targets
- Each line of each file is a sequence, and line i in the inputs files matches line i in the targets file
- The source and target languages are denoted with an abbreviation (e.g. fr and it)
- The names of the files are made of the name of the dataset, the source and target languages and the language of the file, each element being separated by a dot (e.g. MultiCCAligned.fr-it.fr)

The dataset class uses a tokenizer made with the shared vocabulary for both the source and target languages reference corpora. The tokenizer name is that of the dataset and the vocabulary size (e.g. WikiMatrix_10000).

Entries are shuffled with a fixed random state for reproducibility.
Entries are then split in training, validation and testing examples using the number of examples in the dataset and the amount of validation and testing examples provided by the user.
By default, all entries are accessible through the ```getitem``` method. the ```use_set``` method allows to swap between sets ```all```, ```train```, ```val```, ```test```.

The usage of the dataset class is as follows
```python
from dataset import OpusTranslationDataset
from torch.utils.data import DataLoader

# Declare
dataset = OpusTranslationDataset(
        dataset_name="MultiCCAligned.fr-it",
        language_source="fr",
        language_target="it",
        vocab_size=10000,
        sequence_length=128,
        val_examples=1000,
        test_examples=1000
    )

# Swap set (by default all examples are available)
dataset.use_set("train")

# Wrap in dataloader and get first batch
dataloader = DataLoader(dataset, batch_size=8)
batch = next(itet(dataloader))

# Get batched ids sequences and attention masks for sources and targets
source_ids, source_masks, target_ids, target_masks = batch
```

## Transformer
The transformer model implementation is mainly based on the original paper by Vaswani et al [1].
It has two distinct forward methods to be used for training and evaluation. It is built so that it can be trained on any sequence size, with the constraint that the input and target maximal sequence sizes (a.k.a. the length of the attention mask) must be the same.

The first one is used in train mode and produces next tokens using teacher forcing. This means that at each time step, the next token is predicted based on ground truth token instead of the generated ones. In addition, the next token is sampled among the top-10 largest probability tokens. The sampling weights are the probabilities (i.e. after softmax) to the power of 0.2. This is intended to avoid systematically outputting [EOS] tokens (or other really common ones) and thus improve the building of the representation of other tokens.

The second one simply predicts outputs in an autoregressive manner as described in the paper. It uses a greedy decoding procedure as well. It is used when the model is in evaluation mode.

The usage of the transformer module is as follows
```python
from transformer import TransformerModel

# Compute device
device = "cuda:0"

# Model declaration
net = TransformerModel(
    d_model=512, 
    h=8,
    l=6, 
    num_tokens=30000
).to(device=device)

# Making of dummy inputs and targets
sequence_length = 128
batch_size = 4
input_ids = torch.randn(batch_size, sequence_length).to(dtype=torch.int64, device=device)
input_mask = torch.ones(batch_size, sequence_length).to(dtype=torch.float32, device=device)
target_ids = torch.randn(batch_size, sequence_length).to(dtype=torch.int64, device=device)
target_mask = torch.ones(batch_size, sequence_length).to(dtype=torch.float32, device=device)

# Prediction in train mode (with teacher forcing)
net.train()
probs, ids = net(
    input_ids, 
    input_mask,
    target_ids,
    target_mask 
)

# Prediction in evaluation mode (autoregressively)
net.val()
probs, ids = net(
    input_ids, 
    input_mask,
)
```
## References
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.