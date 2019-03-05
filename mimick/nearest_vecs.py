'''
Batch script for obtaining nearest vectors for list of OOV words.
'''

from collections import Counter

import collections
import argparse
import random
import pickle
import math
import codecs
import numpy as np

__author__ = "Yuval Pinter, 2017"

POLYGLOT_UNK = str("<UNK>")
PADDING_CHAR = "<*>"

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

def wordify(instance):
    return ''.join([i2c[i] for i in instance.chars])
    
def dist(v1, v2):
    if options.cosine:
        return 1.0 - (v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.linalg.norm(v1 - v2)
 
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--embs", required=True, dest="embs", help="file with all embeddings")
parser.add_argument("--cosine", dest="cosine", action="store_true", help="Use cosine as diff measure")
parser.add_argument("--words", dest="words_to_check", nargs='*', help="Words to check (random if empty)")
parser.add_argument("--top-to-show", dest="top_to_show", default=10, help="Nearest to show per word")
options = parser.parse_args()

print("Training dataset: {}".format(options.dataset))
print("Embeddings location: {}\n".format(options.embs))

# Load training set
dataset = pickle.load(open(options.dataset, "r"))
c2i = dataset["c2i"]
i2c = { i: c for c, i in list(c2i.items()) } # inverse map
words_to_check = options.words_to_check

test_instances = dataset["test_instances"]
test_words = [wordify(i) for i in test_instances]

if words_to_check is None:
    random.shuffle(test_words)
    words_to_check = test_words[:25]
    
print("Checking words: {}".format(", ".join(words_to_check)))

test_vecs = {}
iv_vecs = {}
with codecs.open(options.embs, "r", "utf-8") as embs_file:
    for line in embs_file:
        split = line.split()
        if len(split) > 2:
            word = split[0]
            vec = np.array([float(s) for s in split[1:]])
            if word in words_to_check:
                test_vecs[word] = vec
            elif not word in test_words:
                iv_vecs[word] = vec

#print "Total in-vocab vecs: {} of size {}".format(len(iv_vecs), len(iv_vecs["the"]))
print("Total test vecs: {}".format(len(test_vecs)))

similar_words = {}
for w, vec in test_vecs.items():
    top_k = sorted([(iv, dist(iv_vec, vec)) for iv,iv_vec in iv_vecs.items()], key=lambda x: x[1])[:options.top_to_show]
    similar_words[w] = top_k

print("\n", "\n".join([k + ":\t" + " ".join([t[0] for t in v]) for k,v in similar_words.items()]))
