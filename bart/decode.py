import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from fairseq.models.lstm import LSTMModel
from fairseq.models.bart import BARTModel
from fairseq.models.transformer import TransformerModel
from fairseq.data.encoders.gpt2_bpe import get_encoder

MODEL_MAP = {
    'lstm': LSTMModel,
    'transformer': TransformerModel,
    'bart': BARTModel,
}


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = MODEL_MAP[args.model_type].from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path
    )

    model.cuda()
    model.eval()

    bpe = None
    if args.encoder_json and args.vocab_bpe:
        bpe = get_encoder(args.encoder_json, args.vocab_bpe)

    num_lines = count_file_lines('{}/test.source'.format(args.data_dir))
    with open('{}/test.source'.format(args.data_dir)) as source, \
            open(args.output_file, 'w') as fout:
        count = 0
        sentences = []
        for line in tqdm(source, total=num_lines):
            sentence = line.strip()
            sentences.append(sentence)
            count += 1
            if count % args.batch_size == 0:
                with torch.no_grad():
                    hypotheses_batch = model.sample(
                        sentences,
                        beam=args.beam_size,
                        lenpen=args.lenpen,
                        max_len_b=args.max_len_b,
                        min_len=args.min_len,
                        no_repeat_ngram_size=args.no_repeat_ngram_size
                    )
                for hypothesis in hypotheses_batch:
                    if args.model_type in ['transformer', 'lstm']:
                        tokens = map(int, hypothesis.split())
                        hypothesis = bpe.decode(tokens)
                    fout.write(hypothesis + '\n')
                    fout.flush()
                sentences = []

        if sentences != []:
            hypotheses_batch = model.sample(
                sentences,
                beam=args.beam_size,
                lenpen=args.lenpen,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )
            for hypothesis in hypotheses_batch:
                if args.model_type in ['transformer', 'lstm']:
                    tokens = map(int, hypothesis.split())
                    hypothesis = bpe.decode(tokens)
                fout.write(hypothesis + '\n')
                fout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help="Model type",
                        choices=['transformer', 'bart', 'lstm'])
    parser.add_argument('--encoder_json', type=str, default=None, help='path to encoder.json')
    parser.add_argument('--vocab_bpe', type=str, default=None, help='path to vocab.bpe')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help="Path of checkpoint directory")
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint_best.pt',
                        help="Path of checkpoint directory")
    parser.add_argument('--data_dir', type=str, required=True, help="Path of data directory")
    parser.add_argument('--data_name_or_path', type=str, required=True, help="Path of the binary data directory")
    parser.add_argument('--output_file', type=str, required=True, help="Path of the output file")
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--max_len_b', type=int, default=60)
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--lenpen', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    main(args)
