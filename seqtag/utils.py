import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import f1_score as intent_f1

from transformers import (
    BertConfig,
    DistilBertConfig,
    AlbertConfig,
    RobertaConfig,
    BertTokenizer,
    DistilBertTokenizer,
    AlbertTokenizer,
    RobertaTokenizer,
)

from seqtag.model import (
    JointBERT,
    JointDistilBERT,
    JointAlbert,
    JointRoberta,
    TransformerConfig,
    JointTransformer,
    TransformerTokenizer,
    RNNConfig,
    JointRNN,
    RNNTokenizer,
    WordTokenizer,
    FeatConfig,
    JointFeatModel
)

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer),
    'roberta': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'transformer': (TransformerConfig, JointTransformer, TransformerTokenizer),
    'rnn': (RNNConfig, JointRNN, RNNTokenizer),
    'rnn-emb': (RNNConfig, JointRNN, WordTokenizer),
    'feature': (FeatConfig, JointFeatModel, WordTokenizer),
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1',
    'roberta': 'roberta-base',
    'transformer': 'transformer',
    'rnn': 'rnn',
    'rnn-emb': 'rnn-emb',
    'feature': 'feature'
}


def get_intent_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def get_postag_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.postag_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = MODEL_CLASSES[args.model_type][2]
    if 'WordTokenizer' in str(tokenizer):
        if not args.vocab_file:
            args.vocab_file = os.path.join(args.data_dir, 'vocab.txt')
            assert os.path.isfile(args.vocab_file)
        return tokenizer(args.vocab_file)
    else:
        return tokenizer.from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(
        intent_preds, intent_labels, slot_preds, slot_labels,
        ignore_intent_label=None, rounding=True
):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_f1(intent_preds, intent_labels)
    exact_match_result = get_sentence_frame_acc(
        intent_preds, intent_labels, slot_preds, slot_labels,
        ignore_intent_label=ignore_intent_label
    )
    if ignore_intent_label:
        keep = [False if i in ignore_intent_label else True for i in intent_labels]
        slot_preds = [i for (i, v) in zip(slot_preds, keep) if v]
        slot_labels = [i for (i, v) in zip(slot_labels, keep) if v]
    slot_result = get_slot_metrics(slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(exact_match_result)

    if rounding:
        for k in results.keys():
            results[k] = np.round(results[k] * 100, 2)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def get_intent_f1(preds, labels):
    return {
        "intent_f1": intent_f1(labels, preds, average='micro')
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(
        intent_preds, intent_labels, slot_preds, slot_labels,
        ignore_intent_label=None
):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    if ignore_intent_label is not None:
        keep = [False if i in ignore_intent_label else True for i in intent_labels]
        intent_result = intent_result[keep]
        slot_result = slot_result[keep]

    exact_match_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "exact_match_acc": exact_match_acc
    }
