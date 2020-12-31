import os
import numpy
import json
import argparse

MODEL_MAP = {
    'rnn': 'BiLSTM',
    'transformer': 'Transformer',
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'rnn + CRF': 'BiLSTM$^{\\ast}$',
    'transformer + CRF': 'Transformer$^{\\ast}$',
    'bert + CRF': 'BERT$^{\\ast}$',
    'roberta + CRF': 'RoBERTa$^{\\ast}$'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument("--model", default='rnn', type=str)
    parser.add_argument("--base_dir", default=None, required=True, type=str, help="Model base dir")
    parser.add_argument("--use_crf", type='bool', default=False, help="CRF-based model")
    parser.add_argument("--seeds", nargs='+', type=int, default=1111)
    args = parser.parse_args()

    tasks = ['entity', 'complex']
    for_latex = True

    suffix = '_crf' if args.use_crf else ''
    model_name = args.model + ' + CRF' if args.use_crf else args.model
    desc = ''

    task_wise_intent_f1 = {t: [] for t in tasks}
    for idx, task in enumerate(tasks):
        slot_f1, em = [], []
        for seed in args.seeds:
            dirname = '{}_{}{}_s{}'.format(task, args.model, suffix, str(seed))
            with open(os.path.join(args.base_dir, dirname, 'eval_results.txt')) as f:
                result = json.load(f)
                task_wise_intent_f1[task].append(result['intent_f1'])
                slot_f1.append(result['slot_f1'])
                em.append(result['exact_match_acc'])

        avg = [
            numpy.round(numpy.mean(slot_f1), 1),
            numpy.round(numpy.mean(em), 1)
        ]
        std = [
            numpy.round(numpy.std(slot_f1), 1),
            numpy.round(numpy.std(em), 1)
        ]
        if for_latex:
            result = ' & '.join(['%.1f$_{\pm %.1f}$' % (a, s) for a, s in zip(avg, std)])
            desc += result
            if idx != len(tasks) - 1:
                desc += ' & '
        else:
            symbol = u"\u00B1"
            result = ';'.join(['%.1f %s %.1f' % (a, symbol, s) for a, s in zip(avg, std)])
            desc += result + ';'

    avg_intent_f1, std_intent_f1 = [], []
    total_count = []
    for task, scores in task_wise_intent_f1.items():
        avg_intent_f1.append(numpy.mean(scores))
        std_intent_f1.append(numpy.std(scores))
        total_count.append(len(scores))

    avg_intent_f1 = numpy.round(numpy.mean(avg_intent_f1), 1)
    value = 0.0
    for v1, v2 in zip(std_intent_f1, total_count):
        value += v1 * v1 * v2
    value = value / sum(total_count)
    std_intent_f1 = numpy.round(numpy.sqrt(value), 1)
    desc = '%.1f$_{\pm %.1f}$' % (avg_intent_f1, std_intent_f1) + ' & ' + desc

    if for_latex:
        desc = MODEL_MAP[model_name] + ' & ' + desc
    else:
        desc = MODEL_MAP[model_name] + '\t' + desc

    if for_latex:
        desc += ' \\\\'
    print(desc)
