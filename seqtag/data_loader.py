import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from seqtag.utils import get_intent_labels, get_slot_labels, get_postag_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(
            self, guid, words, intent_label=None, slot_labels=None, postag_labels=None
    ):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.postag_labels = postag_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids, postag_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids
        self.postag_ids = postag_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.postag_labels = get_postag_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq_{}.out'.format(args.task)
        self.postags_labels_file = 'pos_tags.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, postags, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot, pos_tag) in enumerate(zip(texts, intents, slots, postags)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(
                intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK")
                )
            # 4. pos_tag
            postag_labels = []
            for p in pos_tag.split():
                postag_labels.append(
                    self.postag_labels.index(p) if p in self.postag_labels else self.postag_labels.index("UNK")
                )

            assert len(words) == len(slot_labels) == len(postag_labels)
            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    intent_label=intent_label,
                    slot_labels=slot_labels,
                    postag_labels=postag_labels
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            postags=self._read_file(os.path.join(data_path, self.postags_labels_file)),
            set_type=mode
        )


processors = {
    "type_I": JointProcessor,
    "type_II": JointProcessor,
}


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        label_ids = []  # slot labels
        postag_ids = []  # postag labels
        for word, slot_label, postag_label in zip(example.words, example.slot_labels, example.postag_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            postag_ids.extend([int(postag_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            print('truncate token', len(tokens), max_seq_length, special_tokens_count)
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            postag_ids = postag_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # Add [SEP] token
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        postag_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            postag_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            postag_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            postag_ids = [pad_token_label_id] + postag_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            postag_ids = ([pad_token_label_id] * padding_length) + postag_ids
        else:
            input_ids += ([pad_token] * padding_length)
            attention_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
            postag_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length, \
            "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, \
            "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_length)
        assert len(segment_ids) == max_seq_length, \
            "Error with token type length {} vs {}".format(len(segment_ids), max_seq_length)
        assert len(label_ids) == max_seq_length, \
            "Error with slot labels length {} vs {}".format(len(label_ids), max_seq_length)
        assert len(postag_ids) == max_seq_length, \
            "Error with slot labels length {} vs {}".format(len(postag_ids), max_seq_length)

        intent_label_id = int(example.intent_label)

        assert sum(attention_mask) > 0

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
        #     logger.info("slot_labels: %s" % " ".join([str(x) for x in label_ids]))
        #     logger.info("postag: %s" % " ".join([str(x) for x in postag_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=segment_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=label_ids,
                postag_ids=postag_ids
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "valid":
            examples = processor.get_examples("valid")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, valid, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples,
            args.max_seq_len,
            tokenizer,
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_postag_ids = torch.tensor([f.postag_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask,
        all_token_type_ids, all_intent_label_ids,
        all_slot_labels_ids, all_postag_ids
    )

    return dataset
