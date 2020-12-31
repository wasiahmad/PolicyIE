import re
import json
import argparse
from statistics import mean
from sklearn.metrics import f1_score as intent_f1


def compute_scores(predictions, targets):
    if len(predictions) == 0 and len(targets) == 0:
        return 1.0, 1.0, 1.0

    num_preds = len(predictions)
    num_refs = len(targets)
    num_matches = sum([1 for t in targets if t in predictions])
    p = num_matches / num_preds if num_preds > 0 else 0.0
    r = num_matches / num_refs if num_refs > 0 else 0.0
    f1 = float(2 * (p * r)) / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def load_labels(args):
    intent_labels = []
    entity_labels = []
    complex_labels = []
    with open(args.intent_label) as f:
        for line in f:
            line = line.strip()
            intent_labels.append(line)
    with open(args.entity_arg) as f:
        for line in f:
            line = line.strip()
            if line.startswith('B-'):
                line = line.replace('B-', '')
                entity_labels.append(line)
    with open(args.complex_arg) as f:
        for line in f:
            line = line.strip()
            if line.startswith('B-'):
                line = line.replace('B-', '')
                complex_labels.append(line)

    return intent_labels, entity_labels, complex_labels


def filter_labels(labels, label_list):
    label_suffix = [l.split()[0].replace('arg:', '') for l in labels]
    keep = [True if l in label_list else False for l in label_suffix]
    return [l for idx, l in enumerate(labels) if keep[idx]]


def filter_labels_v2(labels, label_list):
    label_suffix = [l.split()[0].replace('arg:', '') for l in labels]
    keep = [False if l in label_list else True for l in label_suffix]
    return [l for idx, l in enumerate(labels) if keep[idx]]


def fix_arg_labels(labels, label_dict):
    fixed_label = []
    for l in labels:
        l = l.strip()
        if l.startswith('arg : '):
            l = l.replace('arg : ', '')
            for k, v in label_dict.items():
                if l.startswith(k):
                    suffix = l.replace(k, '').strip()
                    m_label = 'arg:' + v + ' ' + suffix
                    fixed_label.append(m_label)
                    break
    return fixed_label


def main(args):
    intent_labels, entity_labels, complex_labels = load_labels(args)
    # lower-case
    intent_labels = [l.lower() for l in intent_labels]
    entity_labels = [l.lower() for l in entity_labels]
    complex_labels = [l.lower() for l in complex_labels]
    label_dict = entity_labels + complex_labels
    label_dict = {' '.join(re.split(r"([.-])", l)): l for l in label_dict}

    int_label_to_id = {k: idx for idx, k in enumerate(intent_labels)}
    pattern = "in:([\w-]+)"
    precision = {'entity': [], 'complex': [], 'all': []}
    recall = {'entity': [], 'complex': [], 'all': []}
    f1_score = {'entity': [], 'complex': [], 'all': []}

    ent_em, com_em, total = 0.0, 0.0, 0.0
    hyp_intents, ref_intents = [], []
    UNK_SYMBOL = 'UNK'.lower()
    OTHER_SYMBOL = 'Other'.lower()
    with open(args.hypotheses) as f1, open(args.references) as f2:
        for hyp, ref in zip(f1, f2):
            # lower-case
            ref = ref.lower()
            hyp, ref = hyp.strip(), ref.strip()
            ref = ref[1:-1]
            if hyp[0] == '[':
                hyp = hyp[1:]
            if hyp[-1] == ']':
                hyp = hyp[:-1]

            hyp_intent = hyp.split('[')[0].strip()
            if hyp_intent.startswith('in : '):
                hyp_intent = hyp_intent.replace('in : ', '')
                hyp_intent = ''.join(hyp_intent.split())
                hyp_intent_id = int_label_to_id.get(hyp_intent, int_label_to_id.get(UNK_SYMBOL))
            else:
                hyp_intent_id = int_label_to_id.get(UNK_SYMBOL)

            ref_intent = re.findall(pattern, ref)
            assert len(ref_intent) == 1
            ref_intent = ref_intent[0]
            ref_intent_id = int_label_to_id.get(ref_intent)

            hyp_intents.append(hyp_intent_id)
            ref_intents.append(ref_intent_id)

            if ref_intent not in [UNK_SYMBOL, OTHER_SYMBOL]:
                hyp_labels = re.findall(r"\[.*?\]", hyp)
                hyp_labels = [l[1:-1] for l in hyp_labels]
                ref_labels = re.findall(r"\[.*?\]", ref)
                ref_labels = [l[1:-1] for l in ref_labels]

                hyp_labels = fix_arg_labels(hyp_labels, label_dict)

                all_p, all_r, all_f1 = compute_scores(hyp_labels, ref_labels)
                precision['all'].append(all_p)
                recall['all'].append(all_r)
                f1_score['all'].append(all_f1)

                # separate labels into entity and complex categories
                # e_hyp_labels = filter_labels(hyp_labels, entity_labels)
                # e_ref_labels = filter_labels(ref_labels, entity_labels)
                e_hyp_labels = filter_labels_v2(hyp_labels, complex_labels)
                e_ref_labels = filter_labels_v2(ref_labels, complex_labels)
                # if e_ref_labels:
                ent_p, ent_r, ent_f1 = compute_scores(e_hyp_labels, e_ref_labels)
                precision['entity'].append(ent_p)
                recall['entity'].append(ent_r)
                f1_score['entity'].append(ent_f1)

                # c_hyp_labels = filter_labels(hyp_labels, complex_labels)
                # c_ref_labels = filter_labels(ref_labels, complex_labels)
                c_hyp_labels = filter_labels_v2(hyp_labels, entity_labels)
                c_ref_labels = filter_labels_v2(ref_labels, entity_labels)
                # if c_ref_labels:
                com_p, com_r, com_f1 = compute_scores(c_hyp_labels, c_ref_labels)
                precision['complex'].append(com_p)
                recall['complex'].append(com_r)
                f1_score['complex'].append(com_f1)

            if ref_intent not in ['UNK', 'Other']:
                total += 1
                # computing exact match
                if hyp_intent_id == ref_intent_id:
                    if ent_f1 == 1.0:
                        ent_em += 1
                    if com_f1 == 1.0:
                        com_em += 1

    result = {}
    int_f1 = intent_f1(hyp_intents, ref_intents, average='micro')
    result["intent_f1"] = round((100.0 * int_f1), 2)
    result["entity_exact_match"] = round((100.0 * ent_em / total), 2)
    result["complex_exact_match"] = round((100.0 * com_em / total), 2)
    for k in ['entity', 'complex', 'all']:
        result["{}_slot_precision".format(k)] = round(100.0 * mean(precision[k]), 2)
        result["{}_slot_recall".format(k)] = round(100.0 * mean(recall[k]), 2)
        result["{}_slot_f1".format(k)] = round(100.0 * mean(f1_score[k]), 2)

    print("***** Eval results *****")
    print(json.dumps(result, indent=4, sort_keys=True))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypotheses', type=str, required=True, help="Path of hypothesis file")
    parser.add_argument('--references', type=str, required=True, help="Path of reference file")
    parser.add_argument('--intent_label', type=str, required=True, help="Path of intent label file")
    parser.add_argument('--entity_arg', type=str, required=True, help="Path of entity arg. file")
    parser.add_argument('--complex_arg', type=str, required=True, help="Path of complex arg. file")
    parser.add_argument('--output_file', type=str, default=None, help="Output file for evaluation results")
    args = parser.parse_args()
    main(args)
