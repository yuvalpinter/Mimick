'''
Run a saved model on a dev or test file.
This file is less refined, documented, and refactored than model.py, but the results are the same. Refer there for documentation.
'''
from __future__ import division
from collections import Counter
from evaluate_morphotags import Evaluator

import collections
import argparse
import random
import cPickle
import logging
import progressbar
import os
import math
import dynet as dy
import numpy as np

import utils

__author__ = "Yuval Pinter, 2017"

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

def get_next_att_batch(attributes, att_tuple):
    ret = {}
    for att in attributes:
        ret[att] = att_tuple.next()
    return ret

class LSTMTagger:

    def __init__(self, rnn_model, use_char_rnn):
        self.use_char_rnn = use_char_rnn

        self.model = dy.Model()
        att_tuple = iter(self.model.load(rnn_model))
        self.attributes = open(rnn_model + "-atts", "r").read().split("\t")
        self.words_lookup = att_tuple.next()
        if (self.use_char_rnn):
            self.char_lookup = att_tuple.next()
            self.char_bi_lstm = att_tuple.next()
        self.word_bi_lstm = att_tuple.next()
        self.lstm_to_tags_params = get_next_att_batch(self.attributes, att_tuple)
        self.lstm_to_tags_bias = get_next_att_batch(self.attributes, att_tuple)
        self.mlp_out = get_next_att_batch(self.attributes, att_tuple)
        self.mlp_out_bias = get_next_att_batch(self.attributes, att_tuple)


    def word_rep(self, w):
        wemb = self.words_lookup[w]
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            char_ids = [pad_char] + [c2i[c] for c in i2w[w]] + [pad_char] # TODO optimize
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            return dy.concatenate([ wemb, char_exprs[-1] ])
        else:
            return wemb


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]

        lstm_out = self.word_bi_lstm.transduce(embeddings)

        H = {}
        Hb = {}
        O = {}
        Ob = {}
        scores = {}
        for att in self.attributes:
            H[att] = dy.parameter(self.lstm_to_tags_params[att])
            Hb[att] = dy.parameter(self.lstm_to_tags_bias[att])
            O[att] = dy.parameter(self.mlp_out[att])
            Ob[att] = dy.parameter(self.mlp_out_bias[att])
            scores[att] = []
            for rep in lstm_out:
                score_t = O[att] * dy.tanh(H[att] * rep + Hb[att]) + Ob[att]
                scores[att].append(score_t)

        return scores


    def tag_sentence(self, sentence):
        observations_set = self.build_tagging_graph(sentence)
        tag_seqs = {}
        for att, observations in observations_set.items():
            observations = [ dy.softmax(obs) for obs in observations ]
            probs = [ obs.npvalue() for obs in observations ]
            tag_seq = []
            for prob in probs:
                tag_t = np.argmax(prob)
                tag_seq.append(tag_t)
            tag_seqs[att] = tag_seq
        return tag_seqs


    def set_dropout(self, p):
        self.word_bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.word_bi_lstm.disable_dropout()

    @property
    def model(self):
        return self.model


def get_att_prop(instances):
    logging.info("Calculating attribute proportions for proportional loss margin")
    total_tokens = 0
    att_counts = Counter()
    for instance in instances:
        total_tokens += len(instance.sentence)
        for att, tags in instance.tags.items():
            t2i = t2is[att]
            att_counts[att] += len([t for t in tags if t != t2i.get(NONE_TAG, -1)])
    return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--model", required=True, dest="model_file", help="Model file to use (.bin)")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Model being read has char RNN trained")
parser.add_argument("--use-dev", dest="use_dev", action="store_true", help="Report on dev set instead of test")
parser.add_argument("--out-dir", default="out", dest="out_dir", help="Directory where to write output")
parser.add_argument("--all-same-col", dest="all_same_col", action="store_true", help="Output examples have POS in same column as tags")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
parser.add_argument("--dynet-seed", dest="dynet_seed", help="Ignore this Dynet param")
options = parser.parse_args()

if options.use_dev:
    devortest = "dev"
else:
    devortest = "test"

# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)
logging.basicConfig(filename=options.out_dir + "/out-{}.txt".format(devortest), filemode="w", format="%(message)s", level=logging.INFO)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logging.info(
"""
Dataset: {}
Using Dev instead of Test: {}
Model input: {}

""".format(options.dataset, options.use_dev, options.model_file))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]
t2is = dataset["t2is"]
c2i = dataset["c2i"]
i2w = { i: w for w, i in w2i.items() } # Inverse mapping
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
i2c = { i: c for c, i in c2i.items() }

tag_lists = { att: [ i2t[idx] for idx in xrange(len(i2t)) ] for att, i2t in i2ts.items() } # To use in the confusion matrix
training_vocab = dataset["training_vocab"]
if options.use_dev:
    test_instances = dataset["dev_instances"]
else:
    test_instances = dataset["test_instances"]

# ===-----------------------------------------------------------------------===
# Load model
# ===-----------------------------------------------------------------------===

tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }

model = LSTMTagger(options.model_file, options.use_char_rnn)

logging.info("Number {} instances: {}".format(devortest, len(test_instances)))

# Evaluate test data
model.disable_dropout()
test_correct = Counter()
test_total = Counter()
test_oov_total = Counter()
bar = progressbar.ProgressBar()
total_wrong = Counter()
total_wrong_oov = Counter()
f1_eval = Evaluator(m = 'att')
if options.debug:
    t_instances = test_instances[0:int(len(test_instances)/10)]
else:
    t_instances = test_instances
with open("{}/{}out.txt".format(options.out_dir, devortest), 'w') as test_writer:
    for instance in bar(t_instances):
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags
        for att in model.attributes:
            if att not in instance.tags:
                gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
        out_tags_set = model.tag_sentence(instance.sentence)

        gold_strings = utils.morphotag_strings(i2ts, gold_tags, not options.all_same_col)
        obs_strings = utils.morphotag_strings(i2ts, out_tags_set, not options.all_same_col)
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
        for att, tags in gold_tags.items():
            out_tags = out_tags_set[att]

            oov_strings = []
            for word, gold, out in zip(instance.sentence, tags, out_tags):
                if gold == out:
                    test_correct[att] += 1
                else:
                    # Got the wrong tag
                    total_wrong[att] += 1
                    if i2w[word] not in training_vocab:
                        total_wrong_oov[att] += 1

                if i2w[word] not in training_vocab:
                    test_oov_total[att] += 1
                    oov_strings.append("OOV")
                else:
                    oov_strings.append("")

            test_total[att] += len(tags)
        test_writer.write(("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings, oov_strings)])
                         + "\n").encode('utf8'))

if options.use_dev:
    logging.info("POS Dev Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
else:
    logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
logging.info("POS % OOV accuracy: {}".format((test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY]))
if total_wrong[POS_KEY] > 0:
    logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
for attr in t2is.keys():
    if attr != POS_KEY:
        logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), options.all_same_col))
logging.info("Total tokens: {}, Total OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))