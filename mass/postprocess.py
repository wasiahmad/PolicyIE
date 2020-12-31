import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, required=True, help="Prefix of the filenames")
    parser.add_argument('--in_file', type=str, required=True, help="Prefix of the filenames")
    parser.add_argument('--out_file', type=str, required=True, help="Path of target directory")
    args = parser.parse_args()

    predictions = {}
    with open(args.index_file) as f1, open(args.in_file) as f2:
        for index, hypo in zip(f1, f2):
            index = int(index.replace('S-', ''))
            hypo = hypo.strip().replace(' ##', '')
            predictions[index] = hypo

    with open(args.out_file, 'w', encoding='utf8') as fw:
        for i in range(len(predictions)):
            fw.write(predictions[i] + '\n')
