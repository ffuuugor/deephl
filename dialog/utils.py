# coding: utf-8
import xml.etree.ElementTree as et
import os
import json
from nltk import word_tokenize, sent_tokenize
from tensorflow.models.rnn.translate.data_utils import initialize_vocabulary
import re
import numpy as np

_DIGIT_RE = re.compile(r"\d")
DATA_DIR = "/Users/ffuuugor/PycharmProjects/googleDeepLearning/data/"
TEXT_PATH = os.path.join(DATA_DIR, "text")
VOCAB_PATH = os.path.join(DATA_DIR, "ffuuu_vocab")
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def deep_files(rootdir):
    for file in os.listdir(rootdir):
        path = os.path.join(rootdir,file)
        if os.path.isdir(path):
            for x in deep_files(path):
                yield x
        else:
            yield path

def gen_dialogs(rootdir=TEXT_PATH):
    for xml in deep_files(rootdir):
        with open(xml,"r") as f:
            for line in f:
                line = line.strip().decode("utf-8")

                if line.startswith("<") or len(line) == 0:
                    continue

                dialog = sent_tokenize(line)
                dialog = map(word_tokenize, dialog)
                yield dialog

def padded(dialog, answer_size=20, dialog_size=20):
    if len(dialog) > dialog_size:
        dialog = dialog[:dialog_size]
    else:
        dialog = dialog + [[_PAD]*answer_size]*(dialog_size - len(dialog))

    for idx, turn in enumerate(dialog):
        if len(turn) > answer_size:
            dialog[idx] = turn[:answer_size]
        else:
            dialog[idx] = turn + [_PAD]*(answer_size - len(turn))

    return dialog

def gen_batches(rootdir=TEXT_PATH, batch_size=2, answer_size=5, dialog_size=4, vocab_path=VOCAB_PATH):
    finished = False
    generator = gen_dialogs(rootdir)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    if dialog_size%2 != 0:
        dialog_size += 1

    while not finished:
        padded_dialogs = []
        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        try:
            for i in range(0, batch_size):
                padded_dialogs.append(padded(next(generator),
                                             answer_size=answer_size, dialog_size=dialog_size))
        except StopIteration:
            finished = True

        for i in range(0, dialog_size):
            data = [
                [padded_dialogs[batch_idx][i][length_idx]
                 for batch_idx in range(0, batch_size)]
                for length_idx in range(0, answer_size)
            ]
            if i%2 == 0:
                encoder_inputs.append(data)
            else:
                data = [[_GO]*batch_size] + data
                decoder_inputs.append(data)

                weights = np.ones((answer_size+1, batch_size), dtype=np.float32)
                for i in range(0, answer_size+1):
                    for j in range(0, batch_size):
                        if data[i][j] == _PAD:
                            weights[i][j] = 0.0

                target_weights.append(weights)

        yield to_ids(encoder_inputs, vocab), to_ids(decoder_inputs, vocab), target_weights


def to_ids(inpt, vocabulary):
    if isinstance(inpt, (str, unicode)):
        return vocabulary.get(re.sub(_DIGIT_RE, "0", inpt), UNK_ID)
    else:
        return map(lambda x: to_ids(x, vocabulary), inpt)





