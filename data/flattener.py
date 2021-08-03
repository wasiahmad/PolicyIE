import argparse


def load_data(dir_path, split):
    data = {}
    with open('{}/{}/seq.in'.format(dir_path, split)) as f:
        for idx, line in enumerate(f):
            data[idx] = {
                'text': line,
                'length': len(line.split())
            }

    with open('{}/{}/seq_type_I.out'.format(dir_path, split)) as f:
        for idx, line in enumerate(f):
            tokens = line.split()
            data[idx]['entity'] = tokens
            assert data[idx]['length'] == len(tokens)

    with open('{}/{}/seq_type_II.out'.format(dir_path, split)) as f:
        for idx, line in enumerate(f):
            tokens = line.split()
            data[idx]['complex'] = tokens
            assert data[idx]['length'] == len(tokens)

    with open('{}/{}/label'.format(dir_path, split)) as f:
        for idx, line in enumerate(f):
            intents = line.strip().split()
            assert len(intents) == 1
            data[idx]['intent'] = intents[0]

    return data


def form_arg_text(tokens, slots):
    arg_tokens, arg_type = [], None
    arg_text = ''
    wrong_slot = 0
    for l in range(len(tokens)):
        if slots[l].startswith('B-'):
            _type = slots[l].replace('B-', '')
            if arg_type is not None and arg_type != _type:
                simple_arg = ' [ARG:{} {}]'.format(arg_type, ' '.join(arg_tokens))
                arg_text += simple_arg
                arg_tokens, arg_type = [], None
            arg_type = _type
            arg_tokens.append(tokens[l])
        elif slots[l].startswith('I-'):
            _type = slots[l].replace('I-', '')
            if arg_type != _type:
                wrong_slot += 1
            else:
                arg_tokens.append(tokens[l])
        else:
            if arg_type:
                simple_arg = ' [ARG:{} {}]'.format(arg_type, ' '.join(arg_tokens))
                arg_text += simple_arg
            arg_tokens, arg_type = [], None

    if arg_type and arg_tokens:
        simple_arg = ' [ARG:{} {}]'.format(arg_type, ' '.join(arg_tokens))
        arg_text += simple_arg

    return arg_text, wrong_slot


def process(dir_path, split):
    with open('{}/{}/seq.out'.format(dir_path, split), 'w') as fw:
        wrong_slot = 0
        data = load_data(dir_path, split)
        for idx in range(len(data)):
            fw.write('[' + 'IN:{}'.format(data[idx]['intent']))
            tokens = data[idx]['text'].split()
            arg_text, ws = form_arg_text(tokens, data[idx]['entity'])
            wrong_slot += ws
            fw.write(arg_text)
            arg_text, ws = form_arg_text(tokens, data[idx]['complex'])
            wrong_slot += ws
            fw.write(arg_text)
            fw.write(']')
            fw.write('\n')

        # print('# wrong slots - ', wrong_slot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="write output to annotations")
    parser.add_argument("--data_dir", required=True,
                        help='Directory path where there is train/test subdirectories')

    args = parser.parse_args()
    process(args.data_dir, 'train')
    process(args.data_dir, 'test')
