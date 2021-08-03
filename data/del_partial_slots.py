import argparse
import os


def main(args):
    seq_id_fn = os.path.join(args.input_dir, "id.in")
    label_fn = os.path.join(args.input_dir, "label")
    seq_in_fn = os.path.join(args.input_dir, "seq.in")
    seq_type_I_fn = os.path.join(args.input_dir, "seq_type_I.out")
    seq_type_II_fn = os.path.join(args.input_dir, "seq_type_II.out")

    with open(seq_in_fn, "r") as f_seq_in, \
            open(seq_type_I_fn, "r") as f_entity, \
            open(seq_type_II_fn, "r") as f_complex, \
            open(label_fn, "r") as f_label, \
            open(seq_id_fn, "r") as f_id:

        seq_id_list = f_id.read().splitlines()
        seq_in_list = f_seq_in.read().splitlines()
        label_list = f_label.read().splitlines()
        seq_type_I_list = f_entity.read().splitlines()
        seq_type_II_list = f_complex.read().splitlines()

    # print(len(seq_id_list), len(seq_in_list), len(label_list), len(seq_type_I_list), len(seq_type_II_list))

    assert len(seq_id_list) == len(seq_in_list) \
           == len(label_list) \
           == len(seq_type_I_list) \
           == len(seq_type_II_list)

    print("Beginning checking partial slots in annotations in the folder ", args.input_dir)

    id_label_2_seq_in = {}
    id_label_2_seq_type_I = {}
    id_label_2_seq_type_II = {}
    for seq_id, seq_in, label, seq_type_I, seq_type_II in zip(seq_id_list,
                                                              seq_in_list,
                                                              label_list,
                                                              seq_type_I_list,
                                                              seq_type_II_list):
        seq_in, seq_type_I, seq_type_II = seq_in.split(" "), seq_type_I.split(" "), seq_type_II.split(" ")
        id_label_2_seq_in["%s_%s" % (seq_id, label)] = seq_in
        id_label_2_seq_type_I["%s_%s" % (seq_id, label)] = seq_type_I
        id_label_2_seq_type_II["%s_%s" % (seq_id, label)] = seq_type_II

    for seq_id, seq_in, label, seq_type_I, seq_type_II in zip(seq_id_list,
                                                              seq_in_list,
                                                              label_list,
                                                              seq_type_I_list,
                                                              seq_type_II_list):
        seq_in, seq_type_I, seq_type_II = seq_in.split(" "), seq_type_I.split(" "), seq_type_II.split(" ")

        # for entity annotation
        if seq_type_I[0].startswith("I-"):
            prev_seq_id = ".".join([seq_id.split(".")[0], seq_id.split(".")[1], str(int(seq_id.split(".")[2]) - 1)])
            # print("Partial slots between %s and %s"% (prev_seq_id, seq_id))

            # handle partial annotation now
            prev_seq_type_I = id_label_2_seq_type_I["%s_%s" % (prev_seq_id, label)]
            assert prev_seq_type_I[-1][2:] == seq_type_I[0][2:]

            # delete the first annotation of the current sequence and last annotation of the previous sequence
            ann_del = seq_type_I[0][2:]
            assert ann_del != ""
            # print("ann_del:", ann_del)
            # print("before deletion,")
            # print("prev: ", prev_seq_type_I)
            # print("curr: ", seq_type_I)

            change_flag = True
            for i in range(len(seq_type_I)):
                if change_flag and seq_type_I[i].endswith(ann_del):
                    seq_type_I[i] = "O"
                    continue
                if not seq_type_I[i].endswith(ann_del):
                    change_flag = False
            change_flag = True
            for i in range(len(prev_seq_type_I) - 1, -1, -1):
                if change_flag and prev_seq_type_I[i].endswith(ann_del):
                    prev_seq_type_I[i] = "O"
                    continue
                if not prev_seq_type_I[i].endswith(ann_del):
                    change_flag = False
            # print("after deletion,")
            # print("prev: ", prev_seq_type_I)
            # print("curr: ", seq_type_I, "\n")

            # update entity seq annotation
            id_label_2_seq_type_I["%s_%s" % (prev_seq_id, label)] = prev_seq_type_I
            id_label_2_seq_type_I["%s_%s" % (seq_id, label)] = seq_type_I

        # for complex annotation
        if seq_type_II[0].startswith("I-"):
            prev_seq_id = ".".join([seq_id.split(".")[0], seq_id.split(".")[1], str(int(seq_id.split(".")[2]) - 1)])
            # print("Partial slots between %s and %s"% (prev_seq_id, seq_id))

            # handle partial annotation now
            prev_seq_type_II = id_label_2_seq_type_II["%s_%s" % (prev_seq_id, label)]
            assert prev_seq_type_II[-1][2:] == seq_type_II[0][2:]

            # delete the first annotation of the current sequence and last annotation of the previous sequence
            ann_del = seq_type_II[0][2:]
            assert ann_del != ""
            # print("ann_del:", ann_del)
            # print("before deletion,")
            # print("prev: ", prev_seq_type_II)
            # print("curr: ", seq_type_II)

            change_flag = True
            for i in range(len(seq_type_II)):
                if change_flag and seq_type_II[i].endswith(ann_del):
                    seq_type_II[i] = "O"
                    continue
                if not seq_type_II[i].endswith(ann_del):
                    change_flag = False
            change_flag = True
            for i in range(len(prev_seq_type_II) - 1, -1, -1):
                if change_flag and prev_seq_type_II[i].endswith(ann_del):
                    prev_seq_type_II[i] = "O"
                    continue
                if not prev_seq_type_II[i].endswith(ann_del):
                    change_flag = False
            # print("after deletion,")
            # print("prev: ", prev_seq_type_II)
            # print("curr: ", seq_type_II, "\n")

            # update complex seq annotation
            id_label_2_seq_type_II["%s_%s" % (prev_seq_id, label)] = prev_seq_type_II
            id_label_2_seq_type_II["%s_%s" % (seq_id, label)] = seq_type_II

    # change label if necessary 
    seq_type_I_list = []
    seq_type_II_list = []
    for i in range(len(seq_id_list)):
        seq_id, label = seq_id_list[i], label_list[i]
        key = "%s_%s" % (seq_id, label)
        seq_type_I = id_label_2_seq_type_I[key]
        seq_type_II = id_label_2_seq_type_II[key]
        if set(seq_type_I) == "O" and set(seq_type_II) == "O" and label != "Other":
            print("Change label for the seq %s, whose the origin label is %s" % (seq_id, label))
            label_list[i] = "Other"

        seq_type_I_list.append(" ".join(seq_type_I))
        seq_type_II_list.append(" ".join(seq_type_II))

    with open(seq_type_I_fn, "w") as f_entity, \
            open(seq_type_II_fn, "w") as f_complex, \
            open(label_fn, "w") as f_label:

        for seq_in, seq_type_I, seq_type_II, label in zip(seq_in_list,
                                                          seq_type_I_list,
                                                          seq_type_II_list,
                                                          label_list):
            assert len(seq_in.split(" ")) == len(seq_type_I.split(" ")) \
                   == len(seq_type_II.split(" "))

            f_entity.write(seq_type_I + "\n")
            f_complex.write(seq_type_II + "\n")
            f_label.write(label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clean the partial slots")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help='input_file')

    args = parser.parse_args()
    main(args)
