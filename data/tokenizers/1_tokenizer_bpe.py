import os
import random
import argparse
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Replace


def train_tokenizer(args) -> None:
    """
    Train a BPE tokenizer for DNA sequences and save it to a file.

    This function sets up and trains a BPE tokenizer using the HuggingFace tokenizers library.
    It configures the tokenizer with specific normalization rules, trains it on a dataset,
    and sets up post-processing with special tokens.

    Args:
        args: Parsed command-line arguments

    Side effects:
    - Sets a random seed for reproducibility
    - Sets an environment variable for tokenizer parallelism
    - Saves the trained tokenizer to a file specified by args.tokenizer_path
    """
    random.seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    print("\nTokenizer training started...\n")

    # Initialize the tokenizer with BPE model
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

    # Configure the trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=args.special_tokens,
        initial_alphabet=["A", "T", "C", "G"],
        max_token_length=args.max_token_length,
    )

    # Train the tokenizer
    tokenizer.train([args.input_file], trainer)

    # Set post-processor with special token references
    tokenizer.post_processor = TemplateProcessing(
        single="[SEP] $A [SEP]",
        pair="[SEP] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
            ("[UNK]", tokenizer.token_to_id("[UNK]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]")),
        ],
    )

    # Save the trained tokenizer
    tokenizer.save(args.tokenizer_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for DNA sequences"
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to the input dataset file"
    )
    parser.add_argument(
        "--tokenizer_path", required=True, help="Path to save the trained tokenizer"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=4096, help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--max_token_length", type=int, default=32, help="Maximum token length"
    )
    parser.add_argument(
        "--special_tokens",
        nargs="+",
        default=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"],
        help="List of special tokens",
    )

    args = parser.parse_args()
    train_tokenizer(args)


if __name__ == "__main__":
    main()
