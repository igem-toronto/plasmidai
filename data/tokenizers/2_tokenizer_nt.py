import pathlib
from typing import Dict

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing


def main() -> None:
    """
    Create and save a simple DNA tokenizer.

    This function creates a basic BPE tokenizer for DNA sequences with a
    vocabulary consisting of the four DNA bases (A, C, G, T). It sets up
    post-processing with special tokens and saves the tokenizer to a JSON file.

    The tokenizer is configured with:
    - A vocabulary of the four DNA bases
    - No merges (each base is treated as a separate token)
    - Special tokens for separation and padding
    - A template for processing single sequences and pairs of sequences

    Side effects:
        Saves the tokenizer to a file named "tokenizer_nt.json" in the same
        directory as this script.
    """
    vocab: Dict[str, int] = {base: i for i, base in enumerate("ACGT")}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[]))

    tokenizer.post_processor = TemplateProcessing(
        single="[SEP] $A [SEP]",
        pair="[SEP] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[SEP]", 4), ("[PAD]", 5)],
    )

    root: pathlib.Path = pathlib.Path(__file__).parent
    tokenizer.save(str(root / "tokenizer_nt.json"))


if __name__ == "__main__":
    main()
