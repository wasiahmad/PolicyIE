import os
import numpy
import json
import argparse

MODEL_MAP = {
    'bart_base': 'BART (base)',
    'bart_large': 'BART (large)',
    'rnn': 'RNN',
    'transformer': 'Transformer'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument("--model", default='rnn', type=str)
    parser.add_argument("--base_dir", default=None, required=True, type=str, help="Model base dir")
    parser.add_argument("--dir_prefix", default=None, required=True, type=str, help="Model dir prefix")
    parser.add_argument("--seeds", nargs='+', type=int, default=1111)
    args = parser.parse_args()

    for_latex = True

    desc = MODEL_MAP[args.model]
    if for_latex:
        desc += ' & '
    else:
        desc += '\t'

    intent_f1, ent_slot_f1, com_slot_f1, ent_em, com_em = [], [], [], [], []
    for seed in args.seeds:
        dirname = '{}-{}-s{}'.format(args.model, args.dir_prefix, str(seed))
        with open(os.path.join(args.base_dir, dirname, 'eval_results.txt')) as f:
            result = json.load(f)
            intent_f1.append(result['intent_f1'])
            ent_slot_f1.append(result['entity_slot_f1'])
            com_slot_f1.append(result['complex_slot_f1'])
            ent_em.append(result['entity_exact_match'])
            com_em.append(result['complex_exact_match'])

    avg = [
        numpy.round(numpy.mean(intent_f1), 1),
        numpy.round(numpy.mean(ent_slot_f1), 1),
        numpy.round(numpy.mean(ent_em), 1),
        numpy.round(numpy.mean(com_slot_f1), 1),
        numpy.round(numpy.mean(com_em), 1)
    ]
    std = [
        numpy.round(numpy.std(intent_f1), 1),
        numpy.round(numpy.std(ent_slot_f1), 1),
        numpy.round(numpy.std(ent_em), 1),
        numpy.round(numpy.std(com_slot_f1), 1),
        numpy.round(numpy.std(com_em), 1)
    ]
    if for_latex:
        result = ' & '.join(['%.1f$_{\pm %.1f}$' % (a, s) for a, s in zip(avg, std)])
        desc += result
    else:
        symbol = u"\u00B1"
        result = ';'.join(['%.1f %s %.1f' % (a, symbol, s) for a, s in zip(avg, std)])
        desc += result + ';'

    if for_latex:
        desc += ' \\\\'
    print(desc)
