import os

from tokenizers import ByteLevelBPETokenizer

def get_tokenizer(tokenizer_path, reference_corpora, vocab_size=30000, sequence_length=256):
    pad_token = "[PAD]"
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[EOT]"]
    if not isinstance(reference_corpora, list):
        reference_corpora = [reference_corpora]
    tokenizer = ByteLevelBPETokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer = tokenizer.from_file(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    else:
        os.makedirs(tokenizer_path)
        tokenizer.train(
        files=reference_corpora, 
        vocab_size=vocab_size, 
        show_progress=True, 
        special_tokens=special_tokens
        )
        tokenizer.save_model(tokenizer_path)
    tokenizer.enable_padding(
        length=sequence_length, 
        pad_id=tokenizer.token_to_id(pad_token), 
        pad_token=pad_token
        )
    tokenizer.enable_truncation(max_length=sequence_length)

    return tokenizer
  
if __name__ == "__main__":
    tokenizer = get_tokenizer("trial_tokenizer", "trials/lorem.txt")
    output = tokenizer.encode("This is an example sentence that is rather short")
    print(output.tokens)