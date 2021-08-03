import argparse

'''
Usage: python python BIO_sanity_check.py --input_file non_overlapping_cross_sentence_level_annotations/train/seq_type_I.out
'''


def sanity_check_bio_file(input_file):
    print("performing BIO sanity check for file %s" % input_file)

    with open(input_file, "r") as fr:
        bio_data = fr.read().splitlines()

    for i, seq in enumerate(bio_data):
        seq = seq.split(" ")
        verify_bio_seq(i, seq)

    print("Passing BIO sanity check for file %s!" % input_file)


def verify_bio_seq(seq_idx, seq):
    for i in range(len(seq)):
        if i == 0:
            # assert seq[i].startswith("B-") or seq[i] == "O", "Error in line %s"% str(seq_idx+1)
            if not (seq[i].startswith("B-") or seq[i] == "O"):
                print("Error in line %s ..." % str(seq_idx + 1))
        if seq[i].startswith("I-") and seq[i - 1] == "O":
            assert ValueError("Wrong format of BIO tagging data.")


def main(args):
    sanity_check_bio_file(args.input_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIO tagging sanity check")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help='input_file')

    args = parser.parse_args()
    main(args)
