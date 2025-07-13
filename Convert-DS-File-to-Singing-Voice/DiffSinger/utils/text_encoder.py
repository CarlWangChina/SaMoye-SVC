import numpy as np

from utils.hparams import hparams

PAD = '<PAD>'
PAD_INDEX = 0


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TokenTextEncoder:
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self, vocab_list):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.

        Args:
            vocab_list: If not None, a list of elements of the vocabulary.
        """
        self.num_reserved_ids = hparams.get('num_pad_tokens', 3)
        assert self.num_reserved_ids > 0, 'num_pad_tokens must be positive'
        self.vocab_list = sorted(vocab_list)

    def encode(self, sentence):
        """Converts a space-separated string of phones to a list of ids."""
        phones = sentence.strip().split() if isinstance(sentence, str) else sentence
        return [self.vocab_list.index(ph) + self.num_reserved_ids if ph != PAD else PAD_INDEX for ph in phones]

    def decode(self, ids, strip_padding=False):
        if strip_padding:
            ids = np.trim_zeros(ids)
        ids = list(ids)
        return ' '.join([
            self.vocab_list[_id - self.num_reserved_ids] if _id >= self.num_reserved_ids else PAD
            for _id in ids
        ])

    def pad(self):
        pass

    @property
    def vocab_size(self):
        return len(self.vocab_list) + self.num_reserved_ids

    def __len__(self):
        return self.vocab_size

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, 'w', encoding='utf8') as f:
            [print(PAD, file=f) for _ in range(self.num_reserved_ids)]
            [print(tok, file=f) for tok in self.vocab_list]
