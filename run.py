"""
Created by davan 
4/21/21
"""
import pandas as pd

import json
import nltk
import os
from DA_Tagger import DATagger
from progress.bar import Bar
import argparse
import torch
import logging
from pprint import pformat

def check_and_create(out_dir):
    if not os.path.exists(out_dir):
        print(" ** creating directory: {}".format(out_dir))
        os.makedirs(out_dir)

def read_lines(fpath):
    with open(fpath, "r") as fin:
        return fin.read().split("\n")

def clean_this_text(text):
    text = str(text).replace('"', " ").strip()
    return text

def process_conversation(conv_turns, tagger):
    act_sequence = []
    act_turn = []
    for j, turn_text in enumerate(conv_turns):
        acts = [clean_this_text(turn_text)]
        act_sequence.extend(acts)
        act_turn.extend([j for _ in acts])

    preds = tagger.tag_conversation(act_sequence, lower=True, sent_tokenize=False)
    assert len(preds) == len(act_sequence)

    turns = []
    current_turn_num = None
    current_turn = {
        "acts": [],
        #"speaker": "A",
    }
    for turn_num, act, pred in zip(act_turn, act_sequence, preds):
        try:
            assert len(pred) == 1, f"len(pred) = {len(pred)} {pred}"
        except Exception as e:
            print(e)
        pred = pred[0]

        if current_turn_num != turn_num:
            turns.append(current_turn)
            current_turn = {
                "acts": [{"text": act, "pred_label": pred["label"], "label_name": pred["label_name"]}],
                #"speaker": "A" if turn_num % 2 == 0 else "B",
            }
            current_turn_num = turn_num
        else:
            current_turn["acts"].append({
                "text": act,
                "pred_label": pred["label"],
                "label_name": pred["label_name"]
            })

    turns.append(current_turn)

    _turns = []
    for j, t in enumerate(turns):
        for a in t["acts"]:
            _turns.append({
                #"speaker": t["speaker"],
                "text": a["text"],
                "pred_label": a["pred_label"],
                "label_name": a["label_name"],
                "turn": j
            })

    return _turns


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, required=True, help="Path of the dataset.")
    parser.add_argument("--conversation_file", type=str, required=True,
                        help="Path of the json file containing conversations.")
    parser.add_argument("--output_path", type=str, required=True,
                        help=".")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path of the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--lower", action="store_true",
                        help="lowercase the data.")

    #args = parser.parse_args([
    #    "--conversation_file", "../DA_Classifier/talk-back/ALL_STUDENT_CODED_+_NON_STUDENT_CODED/conversations-Finished.json",
    #    "--model_path", "../DA_Classifier/models/da-5_acc79.88_loss0.57_e4.pt",
    #    "--lower",
    #    "--output_path", "../DA_Classifier/talk-back/ALL_STUDENT_CODED_+_NON_STUDENT_CODED/conversations-Finished-tagged2.json",
    #])
    args = parser.parse_args()

    logger = logging.getLogger(__file__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    logger.info("torch.cuda.is_available(): " + str(torch.cuda.is_available()))

    with open(args.conversation_file, "r") as fin:
        conversations = json.load(fin)


    config = [
        "-load_model", args.model_path,
    ]
    if args.device == "cuda":
        config.append("-cuda")
    if args.lower:
        config.append("-lower")

    tagger = DATagger(config)
    jj = 0

    for cid, turns in conversations.items():
        print(cid)
        turns = turns["turns"]
        jj += 1
        texts = [t["text"] for t in turns]
        tagged = process_conversation(texts, tagger)
        assert len(tagged) == len(turns), f"len(tagged), len(turns) = {len(tagged)} {len(turns)}"
        for turn, tag in zip(turns, tagged):
            for label in ["pred_label", "label_name"]:
                turn[label] = tag[label]

    print(" * saving", args.output_path)
    with open(args.output_path, "w") as fout:
        json.dump(conversations, fout, indent=2)




if __name__ == "__main__":
    main()

