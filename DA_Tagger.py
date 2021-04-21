


import torch
import torchtext
import torch.nn as nn
import torch.optim as optim

import argparse
import random
from progress.bar import Bar
import dill
import pandas as pd

import json
import nltk
import os
import sys

import logging

try:
    import Dataset
    import train
    import args
except ModuleNotFoundError as e:
    import DA_Classifier.Dataset as Dataset
    import DA_Classifier.train as train
    import DA_Classifier.args as args

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

PAD = Dataset.PAD
UNK = Dataset.UNK
from sklearn.metrics import confusion_matrix


TAG_MAP = {
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
    "Other": "fo_o_fw_by_bc",
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
TAG_MAP = {v: k for k, v in TAG_MAP.items()}

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

def reduce_tagset(swda_all):
    to_be_removed_tags = get_to_remove()

    for i in range(len(swda_all['conversations'])):
        convo = swda_all['conversations'][i]
        for j in range(len(convo['utterances'])):
            if convo['utterances'][j]['damsl_act_tag'] in to_be_removed_tags:
                convo['utterances'][j]['damsl_act_tag'] = 'fo_o_fw_"_by_bc'
                convo['utterances'][j]['act_tag'] = 'fo_o_fw_"_by_bc'
            elif convo['utterances'][j]['damsl_act_tag'] == 'bk':
                convo['utterances'][j]['damsl_act_tag'] = 'b'
                convo['utterances'][j]['act_tag'] = 'b'




def setup_args(arglist=None):
    parser = argparse.ArgumentParser(
        description='eval.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args.add_args(parser)
    if arglist is None:
        opt = parser.parse_args()
    else:
        opt = parser.parse_args(arglist)

    logger.info("Options:" + str(opt))

    return opt


def setup_default_config(cuda=True):

    p1 = "cached_models/m3_acc79.84_loss0.58_e4.pt"
    p2 = "DA_Classifier/cached_models/m3_acc79.84_loss0.58_e4.pt"
    if os.path.exists(p1):
        config = [
            "-load_model", p1,
        ]
    elif os.path.exists(p2):
        config = [
            "-load_model", p2
        ]
    else:
        logger.debug("""Error: could not locate cached model. Please 
        specify a cached model path in the config variable""")


    if cuda:
        config.append("-cuda")

    return config


def load_model(opt):
    assert opt.load_model is not None

    logger.info("Setting up DA model:")
    logger.info(" * loading cached model from {}".format(opt.load_model))
    logger.info(" * torch.cuda.is_available: {}".format(torch.cuda.is_available()))
    if opt.cuda:
        checkpoint = torch.load(opt.load_model, pickle_module=dill)
    else:
        checkpoint = torch.load(opt.load_model, pickle_module=dill, map_location="cpu")
    fields = checkpoint["fields"]
    optim = checkpoint["optim"]
    model_opt = checkpoint["opt"]

    embedding_size = model_opt.word_vec_size

    embeddings = train.build_embeddings(model_opt, fields["conversation"], embedding_size)
    model = train.build_model(model_opt, fields, embeddings)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, fields


class DATagger(object):
    def __init__(self, config=None):

        if config is None:
            config = setup_default_config(False)

        opt = setup_args(config)
        self.model, self.fields = load_model(opt)

        logger.info("Classifier Model:")
        logger.info(str(self.model))

        if opt.cuda:
            self.device = torch.device('cuda:0')
            self.model.to(torch.device('cuda:0'))
        else:
            self.device = None

        self.opt = opt

    def tag_segment(self, segment, lower=False):
        tokens = nltk.word_tokenize(segment)
        if lower:
            tokens = [t.lower() for t in tokens]
        if len(tokens) < 1:
            tokens = [UNK]

        conversation = [[tokens]]

        x, conv_seq,_len, utt_len = self.fields["conversation"].process(conversation, device=self.device)

        preds, crf_input = self.model(x, utt_len)

        prediction = preds[0][0]
        p = self.fields["labels"].vocab.itos[prediction]
        return {
            "label": p,
            "label_name": TAG_MAP.get(p, p)
        }

    def tag_conversation(self, conv, lower=False, sent_tokenize=True):
        """

        :param conv: list of strings. turns of a conversation to be annotated.
        :param lower: [bool] should we lowercase the inputs?
        :return: (conv_turns, preds)
            conv_turns: list of tokenized conversation utterances
            preds: list of DA tags. Aligned with the utterancs of conv_turns.
        """

        # tokenize utterances and preprocess conversation
        conv_sents = []
        sent_turn_map = []
        for j, utt in enumerate(conv):
            turn = []
            sents = [utt]

            if sent_tokenize:
                sents = nltk.sent_tokenize(utt)

            for sent in sents:
                tokens = nltk.word_tokenize(sent)
                if lower:
                    tokens = [t.lower() for t in tokens]
                if len(tokens) < 1:
                    tokens = [UNK]
                sent_turn_map.append(j)
                conv_sents.append(tokens)

        # Process a list of examples to create a torch.Tensor :
        # Pad, numericalize, and postprocess a batch and create a tensor.
        conversation = [conv_sents]
        x, conv_seq_len, utt_len = self.fields["conversation"].process(conversation, device=self.device)

        utt_len = utt_len.to(torch.device("cpu"))

        # run model
        preds, crf_input = self.model(x, utt_len)

        # convert label id's to label names
        #preds = []
        #[self.fields["labels"].vocab.itos[i] for i in preds[0]]

        preds_full_name =  []
        #[TAG_MAP.get(i, i) for i in preds]

        tagged_turns = []
        this_turn = []
        prev_idx = sent_turn_map[0]
        for turn_idx, sent, pred in zip(sent_turn_map, conv_sents, preds[0]):
            if prev_idx != turn_idx:
                tagged_turns.append(this_turn)
                this_turn = []

            p = self.fields["labels"].vocab.itos[pred]
            this_turn.append({
                "tokens": sent,
                "label": p,
                "label_name": TAG_MAP.get(p, p)
            })
            prev_idx = turn_idx
        # add the last turn
        tagged_turns.append(this_turn)

        return tagged_turns


def get_simulated_data(dpath):
    with open(dpath, "r") as fin:
        corpus = json.load(fin)

    examples = []
    for conv in corpus:
        turns = [utt["text"] for utt in conv["utterances"]]
        examples.append(turns)
    """
    examples = [
        [conv1_turn1, conv1_turn2, ...],
        [conv2_turn1, conv2_turn2, ...]
    ]
    """
    return examples


def annotate_split(tagger, split_data, split):

    for conversation_id, conv_data in split_data.items():

        for turn in conv_data["content"]:
            for segment in turn["segments"]:
                da_data = tagger.tag_segment(segment)
                segment["dialog_act"] = da_data

    with open(os.path.join('tc_processed', split + '_anno_full.json'), 'w') as anno_output_file:
        json.dump(split_data, anno_output_file)



def annotate_topical_chats():
    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    tagger = DATagger()

    for split in splits:
        with open(os.path.join('tc_processed', split + '_anno_v2.json'), 'r') as data_file:
            split_data = json.load(data_file)
        annotate_split(tagger, split_data, split)


def main(arglist=None):

    tagger = DATagger()

    data = get_simulated_data("data/dev.json")
    lower = False

    for conv in data:
        conv_turns = tagger.tag_conversation(conv, lower)
        #print("conv_turns:")
        #print(conv_turns)

        for turn in conv_turns:
            #print("turn:", turn)
            turn = turn[0]
            #"tokens": sent,
            #"label": p,
            #"label_name": TAG_MAP.get(p, p)
            print("{} => {}".format(turn["label_name"], " ".join(turn["tokens"])))

        input(">>>")






if __name__ == "__main__":
    ar = [
        "-load_model", "models/m3_acc79.84_loss0.58_e4.pt",
        "-cuda",

    ]

    # main() #ar)
    annotate_topical_chats()



