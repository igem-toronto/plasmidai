import pathlib

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing


def main():
    vocab = {base: i for i, base in enumerate("ACGT")}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[]))

    tokenizer.post_processor = TemplateProcessing(
        single="[SEP] $A [SEP]",
        pair="[SEP] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[SEP]", 4), ("[PAD]", 5)],
    )

    root = pathlib.Path(__file__).parent
    tokenizer.save(str(root / "tokenizer_nt.json"))


if __name__ == '__main__':
    main()
