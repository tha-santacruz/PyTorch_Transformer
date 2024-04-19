import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFKC

def get_tokenizer(tokenizer_path, reference_corpora, vocab_size=30000, sequence_length=256):
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    if not isinstance(reference_corpora, list):
        reference_corpora = [reference_corpora]

    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)

    else:
        trainer = BpeTrainer(
            special_tokens = [unk_token, pad_token, bos_token, eos_token],
            vocab_size=vocab_size,
            min_frequency=2
        )
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
        tokenizer.train(reference_corpora, trainer)
        tokenizer.decoder = ByteLevelDecoder()   
        tokenizer.enable_padding(
            length=sequence_length, 
            pad_id=tokenizer.token_to_id(pad_token), 
            pad_token=pad_token
            )
        tokenizer.enable_truncation(max_length=sequence_length)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, tokenizer.token_to_id(bos_token)),
                (eos_token, tokenizer.token_to_id(eos_token)),
            ]
        )
        tokenizer.normalizer = NFKC()

        tokenizer.save(tokenizer_path)

    return tokenizer
  
if __name__ == "__main__":
    tokenizer = get_tokenizer("trials/trial_tokenizer", "trials/lorem.txt", 30000, 32)
    output = tokenizer.encode("This is an example sentence that is rather short 😀")
    print(output.tokens)