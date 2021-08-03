import os
import argparse


def vocab_process(data_dir):
    slot_label_vocab = 'slot_label.txt'
    intent_label_vocab = 'intent_label.txt'
    postag_label_vocab = 'postag_label.txt'

    train_dir = os.path.join(data_dir, 'train')
    # intent
    with open(os.path.join(train_dir, 'label'), 'r', encoding='utf-8') as f_r, \
            open(os.path.join(data_dir, intent_label_vocab), 'w', encoding='utf-8') as f_w:
        intent_vocab = set()
        for line in f_r:
            line = line.strip()
            intent_vocab.add(line)

        additional_tokens = ["UNK"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        intent_vocab = sorted(list(intent_vocab))
        for intent in intent_vocab:
            f_w.write(intent + '\n')

    # slot
    for suffix in ["type_I", "type_II"]:
        with open(os.path.join(train_dir, 'seq_{}.out'.format(suffix)), 'r', encoding='utf-8') as f_r, \
                open(os.path.join(data_dir, '{}_{}'.format(suffix, slot_label_vocab)), 'w', encoding='utf-8') as f_w:
            slot_vocab = set()
            for line in f_r:
                line = line.strip()
                slots = line.split()
                for slot in slots:
                    slot_vocab.add(slot)

            slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

            # Write additional tokens
            additional_tokens = ["PAD", "UNK"]
            for token in additional_tokens:
                f_w.write(token + '\n')

            for slot in slot_vocab:
                f_w.write(slot + '\n')

    # postag
    with open(os.path.join(train_dir, 'pos_tags.out'), 'r', encoding='utf-8') as f_r, \
            open(os.path.join(data_dir, postag_label_vocab), 'w', encoding='utf-8') as f_w:
        postag_vocab = set()
        for line in f_r:
            line = line.strip()
            postags = line.split()
            for postag in postags:
                postag_vocab.add(postag)

        # Write additional tokens
        additional_tokens = ["PAD", "UNK"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        postag_vocab = sorted(list(postag_vocab))
        for postag in postag_vocab:
            f_w.write(postag + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write output to annotations")
    parser.add_argument("--data_dir", required=True,
                        help='Directory path where there is train/test subdirectories')

    args = parser.parse_args()
    vocab_process(args.data_dir)
