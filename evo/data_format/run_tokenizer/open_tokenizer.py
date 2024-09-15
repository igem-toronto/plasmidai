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

import sentencepiece as spm

def define_tokenizer(ROOT, TOKENIZER):
    return PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER}.json"
    )

if __name__ == '__main__':
    ROOT = '/home/xinleilin/Projects/IGEM/plasmid-ai'
    TOKENIZER = 'dna_bpe_tokenizer_offset'
    sequence = "ATTCTGCGGTTCCCCCTGGAAGACCTACGCAAGTTGGGCCAGCTCAGAGGTGGAATCAACGAAGGCGAGC"
    tokenizer = define_tokenizer(ROOT, TOKENIZER)
    encoded = tokenizer(sequence)
    print("Encoded sequence:", encoded)
    print(type(encoded['input_ids']))  # list of input ids.
    print(encoded['input_ids'])

    # Decode the tokens back to the original sequence
    decoded_sequence = tokenizer.decode(encoded['input_ids'])
    print("Decoded sequence:", decoded_sequence.upper())

    # Show the number of tokens in the tokenizer
    print("Number of tokens in the tokenizer:", tokenizer.vocab_size)
    # number of tokens is 4096!!!!
