from transformers import PreTrainedTokenizerFast


def define_tokenizer(ROOT, TOKENIZER):
    return PreTrainedTokenizerFast(tokenizer_file=f"{TOKENIZER}.json")


if __name__ == "__main__":
    ROOT = "/home/xinleilin/Projects/IGEM/plasmid-ai"
    TOKENIZER = "dna_bpe_tokenizer_offset"
    sequence = "ATTCTGCGGTTCCCCCTGGAAGACCTACGCAAGTTGGGCCAGCTCAGAGGTGGAATCAACGAAGGCGAGC"
    tokenizer = define_tokenizer(ROOT, TOKENIZER)
    encoded = tokenizer(sequence)
    print("Encoded sequence:", encoded)
    print(type(encoded["input_ids"]))  # list of input ids.
    print(encoded["input_ids"])

    # Decode the tokens back to the original sequence
    decoded_sequence = tokenizer.decode(encoded["input_ids"])
    print("Decoded sequence:", decoded_sequence.upper())

    # Show the number of tokens in the tokenizer
    print("Number of tokens in the tokenizer:", tokenizer.vocab_size)
    # number of tokens is 4096!!!!
