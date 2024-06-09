import os
import random
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from Bio import SeqIO

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Normalizer, Replace, Lowercase


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

ROOT = '/scratch/adibvafa/plasmid-ai/'
DATA_ROOT = f'{ROOT}/data'
DATASET = f'{DATA_ROOT}/plasmids.fasta'
DATASET_TXT = f'{DATA_ROOT}/plasmids.txt'
DATASET_DUMMY =f'{DATA_ROOT}/plasmids_dummy.txt'
TOKENIZER = 'dna_bpe_tokenizer'

SEED = 42
VOCAB_SIZE = 4096
NUM_SEQUENCES = 10     #54646
MAX_TOKEN_LENGTH = 32
SPECIAL_TOKENS = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']


if __name__ == '__main__':
    random.seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "1"

    # Train the SentencePiece model with HuggingFace
    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    # Define normalizer
    tokenizer.normalizer = normalizers.Sequence([
        Replace(' ', ''),
        Replace('\n', ''),
        Replace('\r', ''),
        Replace('\t', ''),
    ])

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
        max_token_length=MAX_TOKEN_LENGTH
    )

    # Train tokenizer
    tokenizer.train([DATASET_DUMMY], trainer)

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
        ]
    )

    tokenizer.save(f"{TOKENIZER}.json")