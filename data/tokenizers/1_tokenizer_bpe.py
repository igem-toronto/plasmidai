import os
import random

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Normalizer, Replace, Lowercase


ROOT = '/scratch/adibvafa/plasmid-ai/'
DATA_ROOT = f'{ROOT}/data'
BASE_FILE = f'plasmids_replicon'
DATASET = f'{DATA_ROOT}/{BASE_FILE}.fasta'
DATASET_TXT = f'{DATA_ROOT}/{BASE_FILE}.txt'
DATASET_CUTOFF = f'{DATA_ROOT}/{BASE_FILE}_cutoff.txt'
DATASET_DUMMY =f'{DATA_ROOT}/{BASE_FILE}_dummy.txt'
DATASET_FINETUNE = f'{DATA_ROOT}/{BASE_FILE}_finetune.txt'
DATASET_CUTOFF_RC = f'{DATA_ROOT}/{BASE_FILE}_cutoff_rc.txt'

TOKENIZER = f'{DATA_ROOT}/tokenizer/{BASE_FILE}_dna_bpe_tokenizer_cutoff_rc.json'

SEED = 42
VOCAB_SIZE = 4096
MAX_TOKEN_LENGTH = 32
SPECIAL_TOKENS = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']


if __name__ == '__main__':
    random.seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    print('\nTokenizer training started...\n')

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
    tokenizer.train([DATASET_CUTOFF_RC], trainer)

    # Set post-processor with correct special token references
    tokenizer.post_processor = TemplateProcessing(
        single="[SEP] $A [SEP]",
        pair="[SEP] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
            ("[UNK]", tokenizer.token_to_id("[UNK]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]")),
        ]
    )

    tokenizer.save(TOKENIZER)
