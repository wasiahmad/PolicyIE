import argparse
import json
from tqdm import tqdm


def process(args):
    out_examples = []
    with open(args.source) as f1, open(args.target) as f2:
        for src, tgt in tqdm(zip(f1, f2)):
            out_examples.append({'src': src.strip(), 'tgt': tgt.strip()})
    with open(args.outfile, 'w') as fout:
        fout.write('\n'.join([json.dumps(ex) for ex in out_examples]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help='path to source file')
    parser.add_argument("--target", help='path to target file')
    parser.add_argument("--outfile", help='path to output json file')
    args = parser.parse_args()
    process(args)
