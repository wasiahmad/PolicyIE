import argparse
import os


def main(args):
    seq_id_fn = os.path.join(args.input_dir, "id.in")
    label_fn = os.path.join(args.input_dir, "label")
    seq_in_fn = os.path.join(args.input_dir, "seq.in")
    seq_entity_fn = os.path.join(args.input_dir, "seq_entity.out")
    seq_complex_fn = os.path.join(args.input_dir, "seq_complex.out")

    with open(seq_in_fn, "r") as f_seq_in, \
            open(seq_entity_fn, "r") as f_entity, \
            open(seq_complex_fn, "r") as f_complex, \
            open(label_fn, "r") as f_label, \
            open(seq_id_fn, "r") as f_id:

        seq_id_list = f_id.read().splitlines()
        seq_in_list = f_seq_in.read().splitlines()
        label_list = f_label.read().splitlines()
        seq_entity_list = f_entity.read().splitlines()
        seq_complex_list = f_complex.read().splitlines()

    # print(len(seq_id_list), len(seq_in_list), len(label_list), len(seq_entity_list), len(seq_complex_list))

    assert len(seq_id_list) == len(seq_in_list) \
           == len(label_list) \
           == len(seq_entity_list) \
           == len(seq_complex_list)

    print("Beginning checking partial slots in annotations in the folder ", args.input_dir)

    id_label_2_seq_in = {}
    id_label_2_seq_entity = {}
    id_label_2_seq_complex = {}
    for seq_id, seq_in, label, seq_entity, seq_complex in zip(seq_id_list,
                                                              seq_in_list,
                                                              label_list,
                                                              seq_entity_list,
                                                              seq_complex_list):
        seq_in, seq_entity, seq_complex = seq_in.split(" "), seq_entity.split(" "), seq_complex.split(" ")
        id_label_2_seq_in["%s_%s" % (seq_id, label)] = seq_in
        id_label_2_seq_entity["%s_%s" % (seq_id, label)] = seq_entity
        id_label_2_seq_complex["%s_%s" % (seq_id, label)] = seq_complex

    for seq_id, seq_in, label, seq_entity, seq_complex in zip(seq_id_list,
                                                              seq_in_list,
                                                              label_list,
                                                              seq_entity_list,
                                                              seq_complex_list):
        seq_in, seq_entity, seq_complex = seq_in.split(" "), seq_entity.split(" "), seq_complex.split(" ")

        # for entity annotation
        if seq_entity[0].startswith("I-"):
            prev_seq_id = ".".join([seq_id.split(".")[0], seq_id.split(".")[1], str(int(seq_id.split(".")[2]) - 1)])
            # print("Partial slots between %s and %s"% (prev_seq_id, seq_id))

            # handle partial annotation now
            prev_seq_entity = id_label_2_seq_entity["%s_%s" % (prev_seq_id, label)]
            assert prev_seq_entity[-1][2:] == seq_entity[0][2:]

            # delete the first annotation of the current sequence and last annotation of the previous sequence
            ann_del = seq_entity[0][2:]
            assert ann_del != ""
            # print("ann_del:", ann_del)
            # print("before deletion,")
            # print("prev: ", prev_seq_entity)
            # print("curr: ", seq_entity)

            change_flag = True
            for i in range(len(seq_entity)):
                if change_flag and seq_entity[i].endswith(ann_del):
                    seq_entity[i] = "O"
                    continue
                if not seq_entity[i].endswith(ann_del):
                    change_flag = False
            change_flag = True
            for i in range(len(prev_seq_entity) - 1, -1, -1):
                if change_flag and prev_seq_entity[i].endswith(ann_del):
                    prev_seq_entity[i] = "O"
                    continue
                if not prev_seq_entity[i].endswith(ann_del):
                    change_flag = False
            # print("after deletion,")
            # print("prev: ", prev_seq_entity)
            # print("curr: ", seq_entity, "\n")

            # update entity seq annotation
            id_label_2_seq_entity["%s_%s" % (prev_seq_id, label)] = prev_seq_entity
            id_label_2_seq_entity["%s_%s" % (seq_id, label)] = seq_entity

        # for complex annotation
        if seq_complex[0].startswith("I-"):
            prev_seq_id = ".".join([seq_id.split(".")[0], seq_id.split(".")[1], str(int(seq_id.split(".")[2]) - 1)])
            # print("Partial slots between %s and %s"% (prev_seq_id, seq_id))

            # handle partial annotation now
            prev_seq_complex = id_label_2_seq_complex["%s_%s" % (prev_seq_id, label)]
            assert prev_seq_complex[-1][2:] == seq_complex[0][2:]

            # delete the first annotation of the current sequence and last annotation of the previous sequence
            ann_del = seq_complex[0][2:]
            assert ann_del != ""
            # print("ann_del:", ann_del)
            # print("before deletion,")
            # print("prev: ", prev_seq_complex)
            # print("curr: ", seq_complex)

            change_flag = True
            for i in range(len(seq_complex)):
                if change_flag and seq_complex[i].endswith(ann_del):
                    seq_complex[i] = "O"
                    continue
                if not seq_complex[i].endswith(ann_del):
                    change_flag = False
            change_flag = True
            for i in range(len(prev_seq_complex) - 1, -1, -1):
                if change_flag and prev_seq_complex[i].endswith(ann_del):
                    prev_seq_complex[i] = "O"
                    continue
                if not prev_seq_complex[i].endswith(ann_del):
                    change_flag = False
            # print("after deletion,")
            # print("prev: ", prev_seq_complex)
            # print("curr: ", seq_complex, "\n")

            # update complex seq annotation
            id_label_2_seq_complex["%s_%s" % (prev_seq_id, label)] = prev_seq_complex
            id_label_2_seq_complex["%s_%s" % (seq_id, label)] = seq_complex

    # change label if necessary 
    seq_entity_list = []
    seq_complex_list = []
    for i in range(len(seq_id_list)):
        seq_id, label = seq_id_list[i], label_list[i]
        key = "%s_%s" % (seq_id, label)
        seq_entity = id_label_2_seq_entity[key]
        seq_complex = id_label_2_seq_complex[key]
        if set(seq_entity) == "O" and set(seq_complex) == "O" and label != "Other":
            print("Change label for the seq %s, whose the origin label is %s" % (seq_id, label))
            label_list[i] = "Other"

        seq_entity_list.append(" ".join(seq_entity))
        seq_complex_list.append(" ".join(seq_complex))

    with open(seq_entity_fn, "w") as f_entity, \
            open(seq_complex_fn, "w") as f_complex, \
            open(label_fn, "w") as f_label:

        for seq_in, seq_entity, seq_complex, label in zip(seq_in_list,
                                                          seq_entity_list,
                                                          seq_complex_list,
                                                          label_list):
            assert len(seq_in.split(" ")) == len(seq_entity.split(" ")) \
                   == len(seq_complex.split(" "))

            f_entity.write(seq_entity + "\n")
            f_complex.write(seq_complex + "\n")
            f_label.write(label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clean the partial slots")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help='input_file')

    args = parser.parse_args()
    main(args)
