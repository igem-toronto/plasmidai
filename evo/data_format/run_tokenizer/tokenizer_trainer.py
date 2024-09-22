import os
import random

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Replace


LETTER_TO_BASES = {
    "A": "A",
    "B": "CGT",
    "C": "C",
    "D": "AGT",
    "G": "G",
    "H": "ACT",
    "K": "GT",
    "M": "AC",
    "N": "ACGT",
    "R": "AG",
    "S": "CG",
    "T": "T",
    "V": "ACG",
    "W": "AT",
    "Y": "CT",
}

ROOT = "/scratch/adibvafa/plasmid-ai/"
DATA_ROOT = f"{ROOT}/data"
DATASET = f"{DATA_ROOT}/plasmids.fasta"
DATASET_TXT = f"{DATA_ROOT}/plasmids.txt"
DATASET_DUMMY = f"{DATA_ROOT}/plasmids_dummy.txt"
DATASET_FINETUNE = f"{DATA_ROOT}/plasmids_finetune.txt"
DATASET_CUTOFF = f"{DATA_ROOT}/plasmids_cutoff.txt"
TOKENIZER = f"{DATA_ROOT}/tokenizer/dna_bpe_tokenizer_finetune.json"

SEED = 42
VOCAB_SIZE = 4096
MAX_TOKEN_LENGTH = 32
SPECIAL_TOKENS = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]


if __name__ == "__main__":
    random.seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    print("\nTokenizer training started...\n")

    # Train the SentencePiece model with HuggingFace
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Define normalizer
    tokenizer.normalizer = normalizers.Sequence(
        [
            Replace(" ", ""),
            Replace("\n", ""),
            Replace("\r", ""),
            Replace("\t", ""),
        ]
    )

    # Define the pre-tokenizer
    # pre_tokenizer = pre_tokenizers.Sequence([
    #     Whitespace()
    # ])

    # Train tokenizer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=["A", "T", "C", "G"],
        max_token_length=MAX_TOKEN_LENGTH,
    )

    # Train tokenizer
    tokenizer.train([DATASET_CUTOFF], trainer)

    # Set post-processor with correct special token references
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
            ("[UNK]", tokenizer.token_to_id("[UNK]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]")),
        ],
    )

    tokenizer.save(TOKENIZER)
