import argparse
import sentencepiece as spm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True, type=str, help="The name of input file")
    parser.add_argument("--vocab_size", default=10000, type=int, help="Vocabulary size")
    args = parser.parse_args()

    FILES = [args.input_file]

    spm.SentencePieceTrainer.train(
        '--input={} --vocab_size={} --model_prefix=sentencepiece.bpe '
        '--character_coverage=1.0 --model_type=bpe'.format(
            ','.join(FILES), args.vocab_size
        )
    )
