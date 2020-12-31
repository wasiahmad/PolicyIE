import io
import logging
import collections

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split(' ')[0]
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class WordTokenizer:

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            whitespace_tokenize=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            **kwargs
    ):
        self._unk_token = unk_token
        self._sep_token = sep_token
        self._pad_token = pad_token
        self._cls_token = cls_token
        self._pad_token_type_id = 0
        self.do_lower_case = do_lower_case
        self.whitespace_tokenize = whitespace_tokenize
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def unk_token(self):
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self):
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self):
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self):
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @property
    def unk_token_id(self):
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def vocab_size(self) -> int:
        """ Size of the base vocabulary (without the added tokens) """
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def tokenize(self, text: str, **kwargs):
        if self.whitespace_tokenize:
            tokens = whitespace_tokenize(text)
            if self.do_lower_case:
                tokens = [t.lower() for t in tokens]
            return tokens
        else:
            raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids
