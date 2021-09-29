import json


RELABEL_MAP = {
    "%": "spoken-artifact",
    "+": "+",
    "^2": "spoken-artifact",
    "^g": "tag-question",
    "^h": "spoken-artifact",
    "^q": "statement-opinion",
    "aa": "agree",
    "aap_am": "agree",
    "ad": "action-directive",
    "ar": "disagree",
    "arp_nd": "disagree",
    "b": "backchannel",
    "b^m": "spoken-artifact",
    "ba": "appreciation",
    "bd": "downplayer",
    "bf": "summarize-reformulate",
    "bh": "backchannel-in-question-form",
    "bk": "backchannel",
    "br": "signal-non-understanding",
    "fa": "apology",
    "fc": "conventional-closing",
    "fo_o_fw_\"_by_bc": "backchannel",
    "fp": "conventional-opening",
    "ft": "thanking",
    "h": "hedge",
    "na": "affirmative-non-yes-answer",
    "ng": "negative-non-no-answer",
    "nn": "no-answer",
    "no": "hedge",
    "ny": "yes-answer",
    "oo_co_cc": "offers-options-commits",
    "qh": "rhetorical-question",
    "qo": "open-question",
    "qrr": "or-clause",
    "qw": "wh-question",
    "qw^d": "wh-question",
    "qy": "yes-no-question",
    "qy^d": "yes-no-question",
    "sd": "statement-non-opinion",
    "sv": "statement-opinion",
    "t1": "spoken-artifact",
    "t3": "spoken-artifact",
    "x": "spoken-artifact"
}

def relabel(tag):
    rel = RELABEL_MAP.get(tag)
    if rel is None:
        print(tag)
    return rel



