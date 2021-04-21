"""

Created by davan harrison
email: vharriso@ucsc.edu
Date: 8/19/19

Dataset.py
"""
from __future__ import unicode_literals, print_function, division

import torchtext
from collections import Counter, defaultdict
import pickle
import os
import sys
import json
import dill
import torch
import nltk



PAD = "_PAD_"
UNK = "_UNK_"
PLACE = "_place_"


def check_and_create(out_dir):
    if not os.path.exists(out_dir):
        print(" ** creating directory: {}".format(out_dir))
        os.makedirs(out_dir)


def setup_inference_data(datapath, lower=False):
    print("Preparing data examples:")
    print(" * reading data file:", datapath)
    with open(datapath, "r") as fin:
        corpus = json.load(fin)

    examples = []
    for conv in corpus:
        turns = []
        labels = []

        for utt in conv["utterances"]:
            tokens = nltk.word_tokenize(utt["text"])
            if lower:
                tokens = [t.lower() for t in tokens]
            if len(tokens) < 1:
                tokens = [UNK]
            turns.append(tokens)
            labels.append(utt.get("damsl_act_tag"))

        assert len(turns) == len(labels)
        examples.append((turns, labels))

    return examples


def get_inference_data(data_path, fields):
    raw_examples = setup_inference_data(data_path)
    instance_fields = [("conversation", fields["conversation"]), ("labels", fields["labels"])]
    examples = [torchtext.data.Example.fromlist(conv, instance_fields) for conv in raw_examples]
    inference_data = torchtext.data.Dataset(examples, instance_fields)
    return inference_data, raw_examples



def get_to_remove():
    keep_tags = {
        "Statement-non-opinion": "sd",
        "Acknowledge (Backchannel)": "b",
        "Statement-opinion": "sv",
        "Agree/Accept": "aa",
        # "Abandoned or Turn-Exit": "%",
        "Appreciation": "ba",
        "Yes-No-Question": "qy",
        # "Non-verbal": "x",
        "Yes answers": "ny",
        "Conventional-closing": "fc",
        # "Uninterpretable": "%",
        "Wh-Question": "qw",
        "No answers": "nn",
        "Response Acknowledgement": "bk",
        "Hedge": "h",
        "Declarative Yes-No-Question": "qy^d",
        "Other": 'fo_o_fw_"_by_bc',
        "Backchannel in question form": "bh",
        # "Quotation": "^q",
        "Summarize/reformulate": "bf",
        "Affirmative non-yes answers": "na",
        "Action-directive": "ad",
        # "Collaborative Completion": "^2",
        # "Repeat-phrase": "b^m",
        "Open-Question": "qo",
        "Rhetorical-Questions": "qh",
        "Hold before answer/agreement": "^h",
        "Reject": "ar",
        "Negative non-no answers": "ng",
        "Signal-non-understanding": "br",
        "Other answers": "no",
        "Conventional-opening": "fp",
        "Or-Clause": "qrr",
        # "Dispreferred answers": "arp_nd",
        # "3rd-party-talk": "t3",
        "Offers, Options, Commits": "oo_co_cc",
        # "Self-talk": "t1",
        "Downplayer": "bd",
        "Maybe/Accept-part": "aap_am",
        # "Tag-Question": "^g",
        "Declarative Wh-Question": "qw^d",
        "Apology": "fa",
        "Thanking": "ft",
        # "+": "+"
    }

    all_tags = {
        "Statement-non-opinion": "sd",
        "Acknowledge (Backchannel)": "b",
        "Statement-opinion": "sv",
        "Agree/Accept": "aa",
        "Abandoned or Turn-Exit": "%",
        "Appreciation": "ba",
        "Yes-No-Question": "qy",
        "Non-verbal": "x",
        "Yes answers": "ny",
        "Conventional-closing": "fc",
        "Uninterpretable": "%",
        "Wh-Question": "qw",
        "No answers": "nn",
        "Response Acknowledgement": "bk",
        "Hedge": "h",
        "Declarative Yes-No-Question": "qy^d",
        "Other": 'fo_o_fw_"_by_bc',
        "Backchannel in question form": "bh",
        "Quotation": "^q",
        "Summarize/reformulate": "bf",
        "Affirmative non-yes answers": "na",
        "Action-directive": "ad",
        "Collaborative Completion": "^2",
        "Repeat-phrase": "b^m",
        "Open-Question": "qo",
        "Rhetorical-Questions": "qh",
        "Hold before answer/agreement": "^h",
        "Reject": "ar",
        "Negative non-no answers": "ng",
        "Signal-non-understanding": "br",
        "Other answers": "no",
        "Conventional-opening": "fp",
        "Or-Clause": "qrr",
        "Dispreferred answers": "arp_nd",
        "3rd-party-talk": "t3",
        "Offers, Options, Commits": "oo_co_cc",
        "Self-talk": "t1",
        "Downplayer": "bd",
        "Maybe/Accept-part": "aap_am",
        "Tag-Question": "^g",
        "Declarative Wh-Question": "qw^d",
        "Apology": "fa",
        "Thanking": "ft",
        "+": "+"
    }
    to_be_removed_tags = {
        "Abandoned or Turn-Exit": "%",
        "Non-verbal": "x",
        "Uninterpretable": "%",
        # "Other": 'fo_o_fw_"_by_bc',
        "Quotation": "^q",
        "Collaborative Completion": "^2",
        "Repeat-phrase": "b^m",
        "Dispreferred answers": "arp_nd",
        "3rd-party-talk": "t3",
        "Self-talk": "t1",
        "Tag-Question": "^g",
        "+": "+"
    }
    return set(to_be_removed_tags.values())

def reduce_this_label(in_lab, to_reduce):
    if in_lab in to_reduce:
        retval = 'fo_o_fw_"_by_bc'
    elif in_lab == 'bk':
        retval = 'b'
    else:
        retval = in_lab
    return retval



def setup_data(corpus_path, include_test=False, lower=False):
    print("Preparing data examples:")
    print(" * reading data file:", corpus_path)
    print(" * lowercasing conversation utterances:", lower)
    with open(corpus_path, "r") as fin:
        corpus = json.load(fin)

    print(" * processing examples")
    print(" * {} test examples".format("including" if include_test else "ignoring"))
    all_examples = defaultdict(list)


    to_reduce = get_to_remove()

    for conv in corpus["conversations"]:
        if conv["partition_name"] == "test" and not include_test:
            continue

        turns = []
        labels = []

        for utt in conv["utterances"]:
            tokens = nltk.word_tokenize(utt["text"])
            if lower:
                tokens = [t.lower() for t in tokens]
            if len(tokens) < 1:
                tokens = [UNK]
            turns.append(tokens)
            labels.append(reduce_this_label(utt["damsl_act_tag"], to_reduce))

        assert len(turns) == len(labels)
        all_examples[conv["partition_name"]].append((turns, labels))

    return all_examples


def build_fields(loaded_fields=None):
    if loaded_fields is None:
        LABEL = torchtext.data.Field(sequential=True, unk_token=None, pad_token=None) #, include_lengths=True)
        NESTING_FIELD = torchtext.data.Field(sequential=True,
                                             tokenize=list, unk_token=UNK, pad_token=PAD,
                                             # init_token="<w>", eos_token="</w>"
                                             )

        FIELD = torchtext.data.NestedField(
            NESTING_FIELD,
            #init_token="<s>",
            # eos_token="</s>",
            include_lengths=True,
            # pad_first=True
        )
        fields = {"conversation": FIELD, "labels": LABEL}
    else:
        fields = loaded_fields

    instance_fields = [("conversation", fields["conversation"]),  ("labels", fields["labels"])]
    return fields, instance_fields


def build_examples(input_examples, instance_fields):
    print("Building dataset examples:")

    if isinstance(input_examples, dict):
        examples = {}
        for split_name, conv_list in input_examples.items():
            print(" * constructing {} examples".format(split_name))
            examples[split_name] = [torchtext.data.Example.fromlist(conv, instance_fields) for conv in conv_list]
    else:
        assert isinstance(input_examples, list)
        examples = [torchtext.data.Example.fromlist(conv, instance_fields) for conv in input_examples]

    return examples


def build_dataset(examples, fields, instance_fields, _build_vocabulary=True):
    dataset = {}
    for split_name, exmpl_list in examples.items():
        dataset[split_name] = torchtext.data.Dataset(exmpl_list, instance_fields)


    # build vocabularies
    print("\nProcessing Field vocabularies")
    for name, field_obj in fields.items():
        if _build_vocabulary:
            print("Building vocabulary", name)
            field_obj.build_vocab(dataset["train"], min_freq=1, max_size=100000)
        print(" * {} vocab size:".format(name), len(field_obj.vocab.freqs))
    return dataset


def get_shape(nested_lists):
    s1 = len(nested_lists)
    s2 = len(nested_lists[0])
    s3 = len(nested_lists[0][0])
    return s1, s2, s3


def scratch_pad():
    LABEL = torchtext.data.Field(sequential=True, include_lengths=True)
    NESTING_FIELD = torchtext.data.Field(sequential=True,
                                         tokenize=list, unk_token=UNK, pad_token=PAD,
                                         # init_token="<w>", eos_token="</w>"
                                         )

    field = torchtext.data.NestedField(
        NESTING_FIELD,
        # init_token="<s>",
        # eos_token="</s>",
        include_lengths=True,
        # pad_first=True
    )

    labels = [
        ["aa", "ab", "ac"],
        ["b1", "b2", "b3", "b4"],
        ["c1", "c2"]
    ]

    conv1 = [
        ["oh", ",", "wow",  "."],
        ["you", "'re", "lucky", "."],
        ["that", "'s", "really", "cute", "."]
    ]

    conv2 = [
        ["wow", "."],
        ["oh", ",", "my", "gosh", "."],
        ["oh", ",", "that", "'s", "good", "."],
        ["oh", ",", "is", "n't", "that", "great", "."]
    ]

    conv3 = [
        ["oh", ",", "gosh", "--"],
        ["wow", "."]
    ]

    minibatch = [conv1, conv2, conv3]
    arr, seq_len, words_len = field.pad(minibatch)
    #arr, seq_len = field.pad(minibatch)
    print("arr:")
    print(arr)
    print("\nseq_len:")
    print(seq_len)
    print("\nwords_len:")
    print(words_len)

    print("\nshape:", get_shape(arr))





def get_dataset(opt):
    """

    :param opt:  opt.save_data .load_data .data_path .glove_path
    :param word_vectors:
    :param load_embeddings:
    :param save_embeddings:
    :return:
    """

    out_dir = opt.save_data
    #out_dir = "data"
    check_and_create(out_dir)


    #   Build Fields
    fields_file = os.path.join(out_dir, "fields.dill")
    fields = None
    _build_vocab = True

    print("Checking for preprocessed fields file")
    if os.path.isfile(fields_file):
        if opt.load_data:
            print(" * loading fields file:", fields_file)
            with open(fields_file, "rb") as fin:
                fields = dill.load(fin)
            _build_vocab = False

        else:
            print(" * overwriting existing fields file")

    fields, instance_fields = build_fields(loaded_fields=fields)


    #   Build dataset examples
    examples_file = os.path.join(out_dir, "examples.pt")
    print("Checking for preprocessed examples file")
    if os.path.isfile(examples_file) and opt.load_data:
        print(" * loading examples file:", examples_file)
        examples = torch.load(examples_file)

    else:
        #raw_examples = setup_data("swda/ready_data/swda-corpus.json")
        raw_examples = setup_data(opt.data_path, lower=opt.lower)
        examples = build_examples(raw_examples, instance_fields)

        print(" * saving examples to file:", examples_file)
        torch.save(examples, examples_file)

    #   Build Dataset and Vocabulary
    dataset = build_dataset(examples, fields, instance_fields, _build_vocabulary=_build_vocab)

    print(" * saving fields file:", fields_file)
    with open(fields_file, "wb") as fout:
        dill.dump(fields, fout)

    embeddings = None
    default_emb_path = os.path.join(out_dir, "embeddings.pt")
    if os.path.isfile(default_emb_path) and opt.load_data:
        print("Loading Glove word vectors:")
        print(" * loading vectors from", default_emb_path)
        embeddings = torch.load(default_emb_path)

    if embeddings is None:
        retval = fields, dataset
    else:
        retval = fields, dataset, embeddings

    return retval


def partition_data(corpus_path, save_path):
    print("Preparing data examples:")
    print(" * reading data file:", corpus_path)
    with open(corpus_path, "r") as fin:
        corpus = json.load(fin)

    print(" * processing examples")

    all_examples = defaultdict(list)

    for conv in corpus["conversations"]:
        all_examples[conv["partition_name"]].append(conv)

    for split_name, data_list in all_examples.items():
        fpath = os.path.join(save_path, "{}.json".format(split_name))
        print(" * saving", fpath)
        with open(fpath, "w") as fout:
            json.dump(data_list, fout)





def main():
    #scratch_pad()

    inpath = "swda/ready_data/swda-corpus.json"
    outpath = "data"
    partition_data(inpath, outpath)



if __name__ == "__main__":
    main()
