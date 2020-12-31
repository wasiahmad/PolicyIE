import argparse
import os
import json
import sys
import bisect
import numpy as np
from collections import Counter, defaultdict

import re

ENTITIES_ARGS = {"Data-Collector",
                 "Data-Provider",
                 "Data-Collected",
                 "Data-Sharer",
                 "Data-Receiver",
                 "Data-Shared",
                 "Data-Holder",
                 "Data-Retained",
                 "Retention-Period",
                 "Storage-Place",
                 "Data-Protector",
                 "Data-Protected",
                 "protect-against",
                 }
COMPLEX_ARGS = {"Purpose-Argument",
                "Condition-Argument",
                "Polarity",
                "method",
                }
EVENT_TYPES = {"first-party-collection-use",
               "third-party-sharing-disclosure",
               "data-storage-retention-deletion",
               "data-security-protection"}

''' collect sentence level annotation.
    Usage: python within_sent_annot.py --input_paths ground_true_sanitized_split/train/ --out_dir non_overlapping_cross_sentence_level_annotations/train
           python within_sent_annot.py --input_paths ground_true_sanitized_split/test/ --out_dir non_overlapping_cross_sentence_level_annotations/test
'''


def extract_sentence_level_annotation(json_fn):
    with open(json_fn) as fr:
        data = json.load(fr)

    ###### Step 1 extract non_overlapping segmentation ranges ######
    # use conll format information for more correct sentence boundary.
    text_token = data["words"]
    token_starts = [r[0] for r in data["token-ranges"]]
    token_ends = [r[1] for r in data["token-ranges"]]
    event_annotations = data["event_mentions"]
    sent_token_boundaries = data["sent-token-ranges"]

    pos_tags = data["pos-tags"]
    dependency_relations = data["dependency-relation"]

    ### sanity check ###
    # print("sentence ranges:", sent_token_boundaries)
    ####################

    # find event ranges
    event_ranges = []  # add event ranges
    event_annotations = data["event_mentions"]
    event_id_list = []
    for event in event_annotations:
        min_ann_idx, max_ann_idx = len(text_token) - 1, 0
        trigger = event["trigger"]
        for i, t_idx in enumerate(trigger["token_idx"]):
            # update min_ann_idx and max_ann_idx
            min_ann_idx = min(min_ann_idx, t_idx)
            max_ann_idx = max(max_ann_idx, t_idx)
        for argu in event["arguments"]:
            ann_token_start_idx, ann_token_end_idx = token_starts.index(argu["start_idx"]), token_ends.index(
                argu["end_idx"])
            for i, t_idx in enumerate(range(ann_token_start_idx, ann_token_end_idx + 1)):
                # update min_ann_idx and max_ann_idx
                min_ann_idx = min(min_ann_idx, t_idx)
                max_ann_idx = max(max_ann_idx, t_idx)
        # append the max min ann idx for each event into the event ranges
        event_ranges.append([min_ann_idx, max_ann_idx])
        event_id_list.append(event["event_id"])

    # sort event ranges and event_annotations based on the start index of annotation of events
    zipped_events = sorted(list(zip(event_ranges, event_annotations, event_id_list)), key=lambda x: x[0][0])
    event_ranges = [e_range for e_range, _, _ in zipped_events]
    event_annotations = [e_ann for _, e_ann, _ in zipped_events]
    event_id_list = [e_id for _, _, e_id in zipped_events]

    assert len(event_ranges) == len(event_annotations)
    assert all(
        sent_token_boundaries[i][0] <= sent_token_boundaries[i + 1][0] for i in range(len(sent_token_boundaries) - 1))
    assert all(event_ranges[i][0] <= event_ranges[i + 1][0] for i in range(len(event_ranges) - 1))

    ### sanity check ###
    # print("After sorting")
    # print("sentence ranges:", sent_token_boundaries)
    # print("event ranges:", event_ranges)
    # print("event_id_list", event_id_list)
    ####################

    eventtype_2_entity_arg_annnotations = {}
    eventtype_2_complex_arg_annnotations = {}
    eventtype_2_sent2intentLabel = {event_type: [None] * len(sent_token_boundaries) for event_type in EVENT_TYPES}
    for event_type in EVENT_TYPES:
        entity_arg_annnotations, complex_arg_annnotations = ["O"] * len(text_token), ["O"] * len(text_token)
        for e_id, event, e_range in zip(event_id_list, event_annotations, event_ranges):
            assert e_id == event["event_id"]
            if event_type != event["event_type"]:
                continue
            # clear sequence: one sequence contain only one events for each event types
            # entity_arg_annnotations, complex_arg_annnotations = ["O"]*len(text_token), ["O"]*len(text_token)

            # add event trigger to entity based annotation
            # event_type = event["event_type"]
            trigger = event["trigger"]

            for i, t_idx in enumerate(trigger["token_idx"]):
                assert t_idx >= e_range[0] and t_idx <= e_range[1]
                if i == 0:
                    assert entity_arg_annnotations[t_idx] == "O"
                    entity_arg_annnotations[t_idx] = "B-" + "action"
                else:
                    assert entity_arg_annnotations[t_idx] == "O"
                    entity_arg_annnotations[t_idx] = "I-" + "action"
            # add arguments to entity based annotation or complex argument annotation
            for argu in event["arguments"]:
                argu_role = ''.join([i for i in argu["role"] if not i.isdigit()])
                argu_type = replace_argu_type(argu["type"])
                ann_token_start_idx, ann_token_end_idx = token_starts.index(argu["start_idx"]), token_ends.index(
                    argu["end_idx"])
                if argu_role in ENTITIES_ARGS:
                    if argu_role in {"Data-Shared", "Data-Collected", "Data-Retained", "Data-Protected"}:
                        # data type entity arguments
                        for i, t_idx in enumerate(range(ann_token_start_idx, ann_token_end_idx + 1)):
                            assert t_idx >= e_range[0] and t_idx <= e_range[1]
                            if i == 0:
                                entity_arg_annnotations[t_idx] = "B-" + argu_role.lower() + "." + argu_type.lower()
                            else:
                                entity_arg_annnotations[t_idx] = "I-" + argu_role.lower() + "." + argu_type.lower()
                    else:
                        # non-data type entity arguments
                        for i, t_idx in enumerate(range(ann_token_start_idx, ann_token_end_idx + 1)):
                            assert t_idx >= e_range[0] and t_idx <= e_range[1]
                            if i == 0:
                                entity_arg_annnotations[t_idx] = "B-" + argu_role.lower() + "." + argu_type.lower()
                            else:
                                entity_arg_annnotations[t_idx] = "I-" + argu_role.lower() + "." + argu_type.lower()

                elif argu_role in COMPLEX_ARGS:
                    # TODO: for beautifying the outputs for condition
                    for i, t_idx in enumerate(range(ann_token_start_idx, ann_token_end_idx + 1)):
                        assert t_idx >= e_range[0] and t_idx <= e_range[1]
                        if i == 0:
                            complex_arg_annnotations[t_idx] = "B-" + argu_role.lower() + "." + argu_type.lower()
                        else:
                            complex_arg_annnotations[t_idx] = "I-" + argu_role.lower() + "." + argu_type.lower()
                else:
                    raise NotImplementedError
            # give intent label to each sentence 
            e_start, e_end = e_range[0], e_range[1]
            e_start_sent_id, e_end_sent_id = None, None
            for sent_id, sent_range in enumerate(sent_token_boundaries):
                if e_start >= sent_range[0] and e_start <= sent_range[1]:
                    e_start_sent_id = sent_id
                if e_end >= sent_range[0] and e_end <= sent_range[1]:
                    e_end_sent_id = sent_id
            assert e_start_sent_id is not None and e_end_sent_id is not None

            # print("e_range, e_start_sent_id, e_end_sent_id: ", e_range, e_start_sent_id, e_end_sent_id)
            eventtype_2_sent2intentLabel[event_type][e_start_sent_id:e_end_sent_id + 1] = [1] * (
                    e_end_sent_id + 1 - e_start_sent_id)
        # aggregate annotations
        eventtype_2_entity_arg_annnotations[event_type] = entity_arg_annnotations
        eventtype_2_complex_arg_annnotations[event_type] = complex_arg_annnotations

        ### sanity check ###
        # print("-"*50)
        # print("Event type: ", event_type)
        # print("eventtype_2_sent2intentLabel[event_type]: ", eventtype_2_sent2intentLabel[event_type])
        # print(text_token)
        # print(eventtype_2_entity_arg_annnotations[event_type])
        # print(eventtype_2_complex_arg_annnotations[event_type])
        # print("-"*50)
        ####################

    in_seq_list = []
    entity_arg_out_seq_list = []
    complex_arg_out_seq_list = []
    seq_id_list = []
    label_list = []
    pos_tag_seq_list = []
    dep_rel_seq_list = []
    ann_sent_id = set()
    for event_type in EVENT_TYPES:
        entity_arg_annnotations = eventtype_2_entity_arg_annnotations[event_type]
        complex_arg_annnotations = eventtype_2_complex_arg_annnotations[event_type]
        sent2intentLabel = eventtype_2_sent2intentLabel[event_type]
        # segment the annotation based on sentence ranges
        for sent_id, (ann_flag, seg_range) in enumerate(zip(sent2intentLabel, sent_token_boundaries)):
            # print(sent_id, ann_flag, seg_range)
            if ann_flag == 1:
                seq_start, seq_end = seg_range[0], seg_range[1]
                in_seq = text_token[seq_start:seq_end + 1]
                entity_arg_out_seq = entity_arg_annnotations[seq_start:seq_end + 1]
                complex_arg_out_seq = complex_arg_annnotations[seq_start:seq_end + 1]
                pos_tag_seq = pos_tags[seq_start:seq_end + 1]
                dep_rel_seq = dependency_relations[seq_start:seq_end + 1]
                # add identifier to each sequence
                prefix, base_fn = os.path.split(json_fn)
                sec_id = os.path.splitext(base_fn)[0]
                prefix, policy_name = os.path.split(prefix)
                seq_id = ".".join([policy_name, str(sec_id), str(sent_id)])
                # append list to sequence
                in_seq_list.append(in_seq)
                entity_arg_out_seq_list.append(entity_arg_out_seq)
                complex_arg_out_seq_list.append(complex_arg_out_seq)
                seq_id_list.append(seq_id)
                label_list.append(replace_label(event_type))
                pos_tag_seq_list.append(pos_tag_seq)
                dep_rel_seq_list.append(dep_rel_seq)

    # add unannotated sequence (sentence) to annotation
    # True -> annotated, False -> unannotated
    ann_sent_id_list = or_list(list(eventtype_2_sent2intentLabel.values()))
    assert len(ann_sent_id_list) == len(sent_token_boundaries)
    entity_arg_annnotations, complex_arg_annnotations = ["O"] * len(text_token), ["O"] * len(text_token)
    for sent_id, (ann_flag, seg_range) in enumerate(zip(ann_sent_id_list, sent_token_boundaries)):
        if not ann_flag:
            seq_start, seq_end = seg_range[0], seg_range[1]
            in_seq = text_token[seq_start:seq_end + 1]
            entity_arg_out_seq = entity_arg_annnotations[seq_start:seq_end + 1]
            complex_arg_out_seq = complex_arg_annnotations[seq_start:seq_end + 1]
            pos_tag_seq = pos_tags[seq_start:seq_end + 1]
            dep_rel_seq = dependency_relations[seq_start:seq_end + 1]
            # add identifier to each sequence
            prefix, base_fn = os.path.split(json_fn)
            sec_id = os.path.splitext(base_fn)[0]
            prefix, policy_name = os.path.split(prefix)
            seq_id = ".".join([policy_name, str(sec_id), str(sent_id)])
            # append list to sequence
            in_seq_list.append(in_seq)
            entity_arg_out_seq_list.append(entity_arg_out_seq)
            complex_arg_out_seq_list.append(complex_arg_out_seq)
            seq_id_list.append(seq_id)
            label_list.append("Other")
            pos_tag_seq_list.append(pos_tag_seq)
            dep_rel_seq_list.append(dep_rel_seq)

    # print("-"*50)
    # for in_seq, entity_arg_out_seq, complex_arg_out_seq in zip(in_seq_list, 
    #                                                             entity_arg_out_seq_list, 
    #                                                             complex_arg_out_seq_list):
    #     print("in_seq: ",in_seq)
    #     print("entity_arg_out_seq: ", entity_arg_out_seq)
    #     print("complex_arg_out_seq: ", complex_arg_out_seq)
    #     print("-"*50)

    result = {"in_seq_list": in_seq_list,
              "entity_arg_out_seq_list": entity_arg_out_seq_list,
              "complex_arg_out_seq_list": complex_arg_out_seq_list,
              "label_list": label_list,
              "seq_id_list": seq_id_list,
              "pos_tag_seq_list": pos_tag_seq_list,
              "dep_rel_seq_list": dep_rel_seq_list,
              }

    return result


def or_list(nested_list):
    return [any(l) for l in zip(*nested_list)]


def replace_label(event_type):
    if event_type == "first-party-collection-use":
        return "data-collection-usage"
    if event_type == "third-party-sharing-disclosure":
        return "data-sharing-disclosure"
    else:
        return event_type


def replace_argu_type(argu_type):
    if argu_type == "Adverstising-Marketing":
        return "Advertising-Marketing"
    else:
        return argu_type


def write_to_output(file_prefix, in_seq_list, entity_arg_out_seq_list, complex_arg_out_seq_list, label_list,
                    seq_id_list, pos_tag_seq_list, dep_rel_seq_list):
    print("Writing annotations to outputs ...")
    with open(os.path.join(file_prefix, "seq.in"), "w") as f_seq_in, \
            open(os.path.join(file_prefix, "seq_entity.out"), "w") as f_entity_out, \
            open(os.path.join(file_prefix, "seq_complex.out"), "w") as f_complex_out, \
            open(os.path.join(file_prefix, "label"), "w") as f_label, \
            open(os.path.join(file_prefix, "id.in"), "w") as f_id, \
            open(os.path.join(file_prefix, "pos_tags.out"), "w") as f_pos_tags, \
            open(os.path.join(file_prefix, "dep_rels.out"), "w") as f_dep_rels:
        for in_seq, entity_arg_out_seq, \
            complex_arg_out_seq, label, _id, \
            pos_tag_seq, dep_rel_seq in zip(in_seq_list,
                                            entity_arg_out_seq_list,
                                            complex_arg_out_seq_list,
                                            label_list,
                                            seq_id_list,
                                            pos_tag_seq_list,
                                            dep_rel_seq_list):
            # write_to_file
            f_seq_in.write(" ".join(in_seq) + "\n")
            f_entity_out.write(" ".join(entity_arg_out_seq) + "\n")
            f_complex_out.write(" ".join(complex_arg_out_seq) + "\n")
            f_label.write(label + "\n")
            f_id.write(_id + "\n")
            f_pos_tags.write(" ".join(pos_tag_seq) + "\n")
            f_dep_rels.write(" ".join(dep_rel_seq) + "\n")


def calc_stats(in_seq_list, entity_arg_out_seq_list, complex_arg_out_seq_list, label_list):
    # calculate sequence lengthss
    seq_len_list = [len(in_seq) for in_seq in in_seq_list]
    print("-" * 50)
    print("# of sequence: ", len(seq_len_list))
    print("min avg max sequence length: ", np.min(seq_len_list), np.mean(seq_len_list), np.max(seq_len_list))
    # calculate number of annotation and label information

    label_counter = Counter()
    entity_counter = Counter()
    complex_counter = Counter()
    for in_seq, entity_arg_out_seq, complex_arg_out_seq, label in zip(in_seq_list,
                                                                      entity_arg_out_seq_list,
                                                                      complex_arg_out_seq_list,
                                                                      label_list):

        label_counter[label] += 1
        entity_annotations = [ann[2:] for ann in entity_arg_out_seq if ann.startswith("B-")]
        complex_annotations = [ann[2:] for ann in complex_arg_out_seq if ann.startswith("B-")]

        for ann in entity_annotations:
            entity_counter[ann] += 1
        for ann in complex_annotations:
            complex_counter[ann] += 1

    print("# of intent: ", sum(label_counter.values()))
    print("intent label distribution:", label_counter)
    print()

    print("# of entity annotation: ", sum(entity_counter.values()))
    print("# of unique entity annotation: ", len(entity_counter.values()))
    print(" entity distribution:", entity_counter)

    print()
    print("# of complex annotation: ", sum(complex_counter.values()))
    print("# of unique complex annotation: ", len(complex_counter.values()))
    print("intent complex distribution:", complex_counter)
    print("-" * 50)


def main(args):
    in_seq_list = []
    entity_arg_out_seq_list = []
    complex_arg_out_seq_list = []
    label_list = []
    seq_id_list = []
    pos_tag_seq_list = []
    dep_rel_seq_list = []
    if args.input_paths:
        for folder in args.input_paths:
            for root, _, files in os.walk(folder):
                for fn in files:
                    if fn.endswith(".json"):
                        # print("Parsing annotations for %s" % os.path.join(root, fn))
                        json_fn = os.path.join(root, fn)
                        result = extract_sentence_level_annotation(json_fn)
                        in_seq_list.extend(result["in_seq_list"])
                        entity_arg_out_seq_list.extend(result["entity_arg_out_seq_list"])
                        complex_arg_out_seq_list.extend(result["complex_arg_out_seq_list"])
                        label_list.extend(result["label_list"])
                        seq_id_list.extend(result["seq_id_list"])
                        pos_tag_seq_list.extend(result["pos_tag_seq_list"])
                        dep_rel_seq_list.extend(result["dep_rel_seq_list"])
    if args.input_file:
        # print("Parsing annotations for %s" % args.input_file)
        result = extract_sentence_level_annotation(args.input_file)
        in_seq_list.extend(result["in_seq_list"])
        entity_arg_out_seq_list.extend(result["entity_arg_out_seq_list"])
        complex_arg_out_seq_list.extend(result["complex_arg_out_seq_list"])
        label_list.extend(result["label_list"])
        seq_id_list.extend(result["seq_id_list"])
        pos_tag_seq_list.extend(result["pos_tag_seq_list"])
        dep_rel_seq_list.extend(result["dep_rel_seq_list"])

    assert len(in_seq_list) == len(entity_arg_out_seq_list) \
           == len(complex_arg_out_seq_list) \
           == len(label_list) \
           == len(pos_tag_seq_list) \
           == len(dep_rel_seq_list)
    # print("# of cross_sentence_level_annotations events in total: %d "%len(in_seq_list))

    # write to file
    out_folder = args.out_dir
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    write_to_output(
        file_prefix=out_folder,
        in_seq_list=in_seq_list,
        entity_arg_out_seq_list=entity_arg_out_seq_list,
        complex_arg_out_seq_list=complex_arg_out_seq_list,
        label_list=label_list,
        seq_id_list=seq_id_list,
        pos_tag_seq_list=pos_tag_seq_list,
        dep_rel_seq_list=dep_rel_seq_list
    )
    # calculate stats
    calc_stats(in_seq_list, entity_arg_out_seq_list, complex_arg_out_seq_list, label_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write output to annotations")
    parser.add_argument("--input_paths",
                        # required=True,
                        nargs="+",
                        help='input file paths for annotation')
    parser.add_argument("--input_file",
                        type=str,
                        help='input_file')
    parser.add_argument("--out_dir",
                        type=str,
                        default="non_overlapping_cross_sentence_level_annotations",
                        help='out_dir')

    args = parser.parse_args()
    main(args)
