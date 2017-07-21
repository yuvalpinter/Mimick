'''
Script for extracting a word vector dictionary for a given vocabulary from embedding repository.
Supports Polyglot and W2V formats.
'''
from __future__ import division
import cPickle
import argparse
import codecs
import numpy as np

__author__ = "Yuval Pinter and Robert Guthrie, 2017"

POLYGLOT_UNK = unicode("<UNK>")
W2V_UNK = unicode("UNK")

def read_text_embs(filename):
    words = []
    embs = []
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            split = line.split()
            if len(split) > 2:
                words.append(split[0])
                embs.append(np.array([float(s) for s in split[1:]]))
    return words, embs

def average(embeddings):
    return np.average(embeddings, 0)

def output_word_vector(word, embed, outfile):
    outfile.write(word + " ")
    for i in embed:
        outfile.write(str(i) + " ")
    outfile.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument("--vectors", dest="vectors", help="Pickle file from which to get word vectors")
parser.add_argument("--vocab", dest="vocab", help="File containing words to output embeddings for")
parser.add_argument("--output", dest="output", default="word_and_morpho_embeds.txt", help="Output location")
parser.add_argument("--lowercase-backoff", dest="lowercase_backoff", action="store_true", help="Use lowercased segmentation if not available for capitalized word")
parser.add_argument("--in-vocab-only", dest="in_vocab_only", action="store_true", help="Only output an embedding if it is in-vocab")
parser.add_argument("--average-unk", dest="average_unk", action="store_true", help="Use average embeddings instead of model-given UNK")
parser.add_argument("--w2v", dest="w2v", action="store_true", help="Model is W2V, not Polyglot")
options = parser.parse_args()

# Read in the output vocab
with codecs.open(options.vocab, "r", "utf-8") as f:
    output_words = set([ line.strip() for line in f ])

# read in embeddings
if options.w2v:
    words, embs = read_text_embs(options.vectors)
    unk_sym = W2V_UNK
else:
    words, embs = cPickle.load(open(options.vectors, "r"))
    unk_sym = POLYGLOT_UNK
if options.average_unk:
    unk_emb = average(embs)
else:
    unk_emb = embs[word_to_ix[unk_sym]]
word_to_ix = {w : i for (i,w) in enumerate(words)}

# intersect vocab and embeddings into output file
with codecs.open(options.output, "w", "utf-8") as outfile:
    in_vocab = 0
    total = len(output_words)
    for orig_word in output_words:
        if orig_word not in words and options.lowercase_backoff:
            word = orig_word.lower()
        else:
            word = orig_word
        if word in words:
            embed = embs[word_to_ix[word]]
            output_word_vector(orig_word, embed, outfile)
            in_vocab += 1
        elif options.in_vocab_only: continue
        else:
            embed = unk_emb
            output_word_vector(orig_word, embed, outfile)
    print "Total Number of output words:", total
    print "Total in Training Vocabulary:", in_vocab
    print "Percentage in-vocab:", in_vocab / total
