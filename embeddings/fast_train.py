import fasttext
import argparse
from gensim.models.fasttext import load_facebook_model


def train(args):
    outfile = "polisis_fast_embeddings_{}.bin".format(args.emb_size)

    #  -dim 300 -wordNgrams 3 -ws 2 -epoch 200 -verbose 3
    model = fasttext.train_unsupervised(
        args.input_file,
        model='skipgram',
        dim=args.emb_size,
        wordNgrams=3,
        ws=2,
        epoch=200,
        verbose=3
    )
    model.save_model(outfile)


def save(args):
    infile = "polisis_fast_embeddings_{}.bin".format(args.emb_size)
    outfile = "polisis_fast_embeddings_{}.txt".format(args.emb_size)

    fb_model = load_facebook_model(infile)
    fb_model.wv.save_word2vec_format(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True, type=str, help="The name of input file")
    parser.add_argument("--emb_size", default=100, type=int, help="Embedding size")
    args = parser.parse_args()

    train(args)
    save(args)
