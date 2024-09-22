import torch
from torch.utils.data import Dataset
import os
import csv
from Bio import SeqIO

from data_format.tokenizer.open_tokenizer import define_tokenizer
import torch.nn.functional as F

# for default evo fine_tune
from stripedhyena.tokenizer import CharLevelTokenizer


class PlasmidDataset(Dataset):
    def __init__(self, config, train_set, real_job, tokenizer=None):
        self.config = config
        self.splits_path = self.config["splits_file"]
        self.data_path = self.config["data_path"]
        self.train_set = train_set
        self.length_seq = self.config["max_length"]
        self.real_job = real_job
        self.full_debug = self.config["full_debug"]
        self.length_subsequence = self.config["length_seq"]

        self.train_name, self.test_name = self.read_splits(
            self.splits_path, self.real_job
        )
        self.sequences = self.read_sequences(
            self.data_path, self.length_seq, self.train_set
        )

        if self.config["use_custom_tokens"]:
            self.tokenizer = define_tokenizer(
                self.config["root"],
                os.path.join(
                    os.getcwd(), "data_format/tokenizer", self.config["tokenizer"]
                ),
            )
        else:
            # a tokenizer with vocab size 512. (which is also the size of the logits in evo)
            # self.tokenizerObject = CharLevelTokenizer(512).tokenizer
            self.tokenizer = CharLevelTokenizer(512).tokenize

        if self.tokenizer is None:
            self.tokenizer = self.my_tokenizer

        # print('self.sequences', self.sequences)
        self.treated_sequences = []
        self.masks = []
        self.outputs = []
        # now make the inputs for the model, by having only the given sequence length
        for sequence in self.sequences:
            # only take the sequence
            sequence = sequence[3]

            # apply transformations to the dna
            dna = self.random_circular_crop(sequence, Lmax=self.length_seq)

            val = self.tokenizer(dna)  # val is a list

            dna = torch.tensor(val, dtype=torch.float32)

            # create the mask, although not using yet.
            mask = torch.full(dna.shape, True)

            # Padding to Lmax + 1
            pad = (self.length_seq + 1) - dna.shape[0]
            dna = F.pad(dna, pad=(0, pad))
            mask = F.pad(mask, pad=(0, pad), value=False)

            self.treated_sequences.append(dna)
            self.masks.append(mask)

        print(
            f"Train is set to {self.train_set}, and the number of sequences is {len(self.treated_sequences)}"
        )
        self.treated_sequences = torch.stack(self.treated_sequences)
        self.masks = torch.stack(self.masks)
        # self.outputs = torch.stack(self.outputs)

    def __len__(self):
        return len(self.treated_sequences)

    def __getitem__(self, idx):
        sequence = self.treated_sequences[idx]  # get the sequence
        mask = self.masks[idx]
        # note: need to make the input into long, so each token wil become a int.
        sequence = torch.tensor(sequence, dtype=int)
        return (
            sequence,
            mask,
        )  # , output_label # mask says where the input was truncated

    def random_circular_crop(self, dna, Lmax):
        """implementation from
        https://github.com/igem-toronto/plasmid-ai/blob/e4a5352f1aa660b4f0a14bcd10e3f946d854a5dd/src/utils.py#L218
        """
        start = torch.randint(len(dna), size=[1]).item()
        L = min(len(dna), Lmax)
        crop = dna[start : (start + L)]
        overhang = L - len(crop)  # wrap around to start
        crop = crop + dna[:overhang]
        assert len(crop) == L
        return crop

    def read_splits(self, splits_path, real_job):
        train_ids = set()
        test_ids = set()

        with open(splits_path, "r") as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if (len(train_ids) > 5 or len(test_ids) > 5) and not real_job:
                    break

                if row["split"] == "train" and self.train_set:
                    train_ids.add(str(row["id"]))

                if row["split"] == "test" and not self.train_set:
                    test_ids.add(str(row["id"]))

        print("The length of the training ids", len(train_ids))
        print("The length of the testing ids", len(test_ids))

        if self.full_debug or not self.real_job:
            print("Train IDs:", train_ids)
            print("Test IDs:", test_ids)

        return train_ids, test_ids

    # TODO fix the procedure by which you mask the sequences... I don't think that's the correct way.
    def masking_sequence(self, tokenized_sequence):
        mask_position = torch.randint(1, len(tokenized_sequence), (1,)).item()

        # I will let the model predict what the sequence at 9 is
        # then remove everything after the masked token, to let the model generate the rest.
        masked_sequence = tokenized_sequence[
            :mask_position
        ]  # + [9] #+ tokenized_sequence[mask_position + 1:]
        hidden_token = tokenized_sequence[mask_position]

        return masked_sequence, hidden_token

    def read_sequences(self, data_path, length_seq, train_set):
        # I'll skip the sequences that are too long... its taking to much meomry
        records = [
            r for r in SeqIO.parse(data_path, "fasta") if len(r.seq) < length_seq
        ]

        if self.full_debug:
            print("The length of record is:", len(records))

        record_dict = []
        for record in records:
            record_data = [
                record.id,
                record.name,
                record.description,
                str(record.seq),
                record.annotations,
                record.letter_annotations,
            ]

            if train_set and str(record_data[0]) in self.train_name:
                record_dict.append(record_data)

            if not train_set and str(record_data[0]) in self.test_name:
                record_dict.append(record_data)

        return record_dict
