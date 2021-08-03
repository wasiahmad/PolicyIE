import io
import os
import argparse
from collections import Counter


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data, n, d


def form_vocab(dirpath):
    vocab = list()
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(dirpath, split, 'seq.in')) as f:
            for line in f:
                tokens = line.strip().split()
                vocab.extend(tokens)
    vocab = Counter(vocab)
    return vocab


def filter_vectors(vectors, vocab):
    data = {}
    for w in vocab:
        if w in vectors:
            data[w] = vectors[w]
    return data


def save_vectors(vectors, d, fname):
    with open(fname, 'w', encoding='utf-8') as fw:
        fw.write(str(len(vectors)) + ' ' + str(d) + '\n')
        for k, v in vectors.items():
            fw.write(k + ' ' + ' '.join(map(str, v)) + '\n')


def save_vocab(fname, vocab):
    with open(fname, 'w', encoding='utf-8') as fw:
        fw.write('[PAD] 0\n')
        fw.write('[UNK] 0\n')
        fw.write('[CLS] 0\n')
        fw.write('[SEP] 0\n')
        for k, v in vocab.most_common():
            fw.write(k + ' ' + str(v) + '\n')


def main(args):
    vocab = form_vocab(args.data_dir)
    save_vocab(args.vocab_file, vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory path of the data")
    parser.add_argument('--vocab_file', type=str, required=True, help="Filepath of the output vocab file")
    args = parser.parse_args()
    main(args)
