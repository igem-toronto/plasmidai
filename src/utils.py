from typing import Union, List, Optional

import torch
from transformers import PreTrainedTokenizerFast


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


class PlasmidTokenizer:

    def __init__(self):
        self.vocab = dict()
        self.eos = "eos"

        # Regular tokens
        self.index_to_base = "ACGT"
        for base in self.index_to_base:
            self._register_token(base)
        self._register_token(self.eos)

    def _register_token(self, name):
        self.vocab[name] = len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, dna, sos=True, eos=True):
        tokens = []
        if sos:
            tokens.append(self.eos)
        for x in dna:
            choices = LETTER_TO_BASES[x]
            i = torch.randint(len(choices), size=[1]).item()
            tokens.append(choices[i])
        if eos:
            tokens.append(self.eos)
        sequence = [self.vocab[tok] for tok in tokens]
        return torch.tensor(sequence, dtype=torch.long)

    def decode(self, sequence):
        assert sequence.ndim == 1
        assert torch.all((0 <= sequence) & (sequence < self.vocab_size))

        dna = []
        for idx in sequence:
            idx = idx.item()
            if idx == self.vocab[self.eos]:
                break
            dna.append(self.index_to_base[idx])
        return "".join(dna)


class PlasmidBPETokenizer:
    """
    A tokenizer class for handling DNA sequences using Byte Pair Encoding (BPE).
    
    Attributes:
        tokenizer_path (str): Path to the tokenizer file.
        tokenizer (PreTrainedTokenizerFast): Loaded tokenizer object.
        special_tokens (dict): Dictionary containing special token mappings.
    """

    def __init__(self, tokenizer_path: str):
        """
        Initializes the PlasmidBPETokenizer with the provided tokenizer path.

        Args:
            tokenizer_path (str): Path to the tokenizer file.
        """
        # Load tokenizer from file
        self.tokenizer_path = tokenizer_path
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)

        # Define special tokens
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'

        # Create a dictionary of the special tokens
        self.special_tokens = {
            'cls_token': self.cls_token,
            'sep_token': self.sep_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'mask_token': self.mask_token
        }

        # Add special tokens to the tokenizer
        self.tokenizer.add_special_tokens(self.special_tokens)

    def vocab_size(self) -> int:
        """
        Returns the size of the tokenizer's vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.tokenizer.vocab_size
    
    def get_vocab(self) -> dict:
        """
        Returns the tokenizer's vocabulary.

        Returns:
            dict: The tokenizer's vocabulary.
        """
        return self.tokenizer.vocab
    
    def tokenize(
        self,
        dna: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = True,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
        return_tensors: str = 'pt',
        only_input_ids: bool = True
    ) -> Union[dict, List[int]]:
        """
        Tokenizes a DNA sequence.

        Args:
            dna (str): The DNA sequence to tokenize.
            max_length (Optional[int]): Maximum length of the tokenized sequence.
            truncation (bool): Whether to truncate the sequence.
            padding (bool): Whether to pad the sequence.
            return_attention_mask (bool): Whether to return the attention mask.
            return_token_type_ids (bool): Whether to return token type IDs.
            return_tensors (str): The type of tensors to return.
            only_input_ids (bool): Whether to return only input IDs.

        Returns:
            Union[dict, List[int]]: Tokenized output or input IDs.
        """
        output = self.tokenizer(
            dna,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=return_tensors
        )
        return output if not only_input_ids else output['input_ids']
    
    def decode(
        self,
        sequence: Union[List[int], str],
        skip_special_tokens: bool = False,
        truncate: bool = True
    ) -> str:
        """
        Decodes a tokenized sequence back to a string.

        Args:
            sequence (Union[List[int], str]): The tokenized sequence to decode.
            skip_special_tokens (bool): Whether to skip special tokens.
            truncate (bool): Whether to truncate the output at the first [SEP] token.

        Returns:
            str: The decoded string.
        """
        output = self.tokenizer.decode(
            sequence,
            skip_special_tokens=skip_special_tokens,
        )

        # If truncate, truncates at the first [SEP] token and adds the [SEP] back
        return output if not truncate else output.split(self.sep_token)[0] + self.sep_token

    def convert_tokens_to_ids(self, token_sequence: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a sequence of tokens to their corresponding IDs.

        Args:
            token_sequence (Union[str, List[str]]): The token sequence to convert.

        Returns:
            Union[int, List[int]]: The corresponding token IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(token_sequence)
    
    def convert_ids_to_tokens(self, id_sequence: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Converts a sequence of IDs to their corresponding tokens.

        Args:
            id_sequence (Union[int, List[int]]): The ID sequence to convert.

        Returns:
            Union[str, List[str]]: The corresponding tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(id_sequence)


def random_circular_crop(dna, Lmax):
    start = torch.randint(len(dna), size=[1]).item()
    L = min(len(dna), Lmax)
    crop = dna[start:(start + L)]
    overhang = L - len(crop)  # wrap around to start
    crop = crop + dna[:overhang]
    assert len(crop) == L
    return crop
