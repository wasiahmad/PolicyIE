import argparse
import os

'''
   Usage: python shorten_labels.py --input_path non_overlapping_cross_sentence_level_annotations/train --out_dir non_overlapping_cross_sentence_level_annotations/train
          python shorten_labels.py --input_path non_overlapping_cross_sentence_level_annotations/test --out_dir non_overlapping_cross_sentence_level_annotations/test
'''

Ann2short = {'data-user-online-activities-profiles': "D-UOAP",
             'data-general': "D-Gen",
             'data-cookies-web-beacons-and-other-technologies': "D-CWBAOT",
             'data-sharer': "DS-er",
             'advertising-marketing': "AM",
             'condition': "Cond",
             'first-party-entity': "1st-P",
             'polarity': "Pol",
             'legal-requirement': "LR",
             'data-protector': "DPro-er",
             'data-location': "D-Loc",
             'storage-place': "SP",
             'data-retained': "DR",
             'data-computer-device': "D-CD",
             'merge-acquisition': "MA",
             'method': "Meth",
             'communications': "Comm",
             'data-aggregated-nonidentifiable': "D-AN",
             'analytics-research': "AR",
             'user-authentication': "UA",
             'data-finance': "D-Fin",
             'personalization-customization': "PC",
             'data-contact': "D-Con",
             'other': "Pur-Oth",
             'encryptions': "Enc",
             'protect-against': "PA",
             'condition-argument': "Cond-ARG",
             'data-protected': "DP",
             'user': "Usr",
             'security-threat': "ST",
             'service-operation-and-security': "SOAS",
             'retention-period': "RP",
             'data-shared': "DS",
             'data-receiver': "DR-er",
             'action': "Act",
             'access-limitation': "AL",
             'protection-other': "Pro-Oth",
             'general-safeguard-method': "GSM",
             'data-provider': "DPer",
             'purpose-argument': "Pur-ARG",
             'data-collector': "DC-er",
             'basic-service-feature': "BSF",
             'third-party-entity': "3rd-P",
             'data-holder': "DH-er",
             'data-other': "D-Oth",
             'negation': "Neg",
             'data-collected': "DC",
             'data-demographic': "D-Dem"
             }


def shorten_ann(ann):
    if ann == "O":
        return ann

    prefix, role_type = ann[:2], ann[2:]
    if "." in role_type:
        role, _type = role_type.split(".")
        ann = prefix + Ann2short[role] + "." + Ann2short[_type]
    else:
        role = role_type
        ann = prefix + Ann2short[role]

    return ann


def main(args):
    with open(os.path.join(args.input_path, "seq_entity.out"), "r") as f_entity_out, \
            open(os.path.join(args.input_path, "seq_complex.out"), "r") as f_complex_out:

        entity_out_seq_list = f_entity_out.read().splitlines()
        complex_out_seq_list = f_complex_out.read().splitlines()

    assert len(entity_out_seq_list) == len(complex_out_seq_list)

    short_entity_out_seq_list = []
    short_complex_out_seq_list = []
    for entity_out_seq_str, complex_out_seq_str in zip(entity_out_seq_list, complex_out_seq_list):
        entity_out_seq, complex_out_seq = entity_out_seq_str.split(" "), complex_out_seq_str.split(" ")
        short_entity_out_seq = []
        short_complex_out_seq = []
        assert len(entity_out_seq) == len(complex_out_seq)
        for entity_ann, complex_ann in zip(entity_out_seq, complex_out_seq):
            short_entity_ann, short_complex_ann = shorten_ann(entity_ann), shorten_ann(complex_ann)
            short_entity_out_seq.append(short_entity_ann)
            short_complex_out_seq.append(short_complex_ann)

        short_entity_out_seq_list.append(" ".join(short_entity_out_seq) + "\n")
        short_complex_out_seq_list.append(" ".join(short_complex_out_seq) + "\n")

    with open(os.path.join(args.out_dir, "short_seq_entity.out"), "w") as f_entity_out, \
            open(os.path.join(args.out_dir, "short_seq_complex.out"), "w") as f_complex_out:

        f_entity_out.writelines(short_entity_out_seq_list)
        f_complex_out.writelines(short_complex_out_seq_list)
    #     for short_entity_out_seq_str, short_complex_out_seq_str in zip(short_entity_out_seq_list, short_complex_out_seq_list): 
    #         f_entity_out.write(short_entity_out_seq_str+"\n")
    #         f_complex_out.write(short_complex_out_seq_str+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shorten label")
    parser.add_argument("--input_path",
                        required=True,
                        help='input file paths for annotation')

    parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        default="non_overlapping_cross_sentence_level_annotations",
                        help='out_dir')

    args = parser.parse_args()
    ann_val_list = list(Ann2short.values())
    assert len(set([x for x in ann_val_list if ann_val_list.count(x) > 1])) == 0
    main(args)
