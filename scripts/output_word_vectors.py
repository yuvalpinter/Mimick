'''
Script for extracting a word vector dictionary for a given vocabulary from embedding repository.
Supports Polyglot and W2V formats.
'''

import pickle
import argparse
import codecs
import numpy as np

__author__ = "Yuval Pinter and Robert Guthrie, 2017"

POLYGLOT_UNK = str("<UNK>")
W2V_UNK = str("UNK")

def read_text_embs(files):
    word_embs = dict()
    for filename in files:
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                split = line.split()
                if len(split) > 2:
                    word_embs[split[0]] = np.array([float(s) for s in split[1:]])
    return list(zip(*word_embs.items()))

def read_pickle_embs(files):
    word_embs = dict()
    for filename in files:
        print(filename)
        #this might need to be a "rb" read mode
        words, embs = pickle.load(open(filename, "r"))
        word_embs.update(list(zip(words, embs)))
    return list(zip(*word_embs.items()))

def average(embeddings):
    return np.average(embeddings, 0)

def output_word_vector(word, embed, outfile):
    outfile.write(word + " ")
    for i in embed:
        outfile.write(str(i) + " ")
    outfile.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument("--vectors", required=True, nargs="*", dest="vectors", help="Pickle file(s) from which to get word vectors")
parser.add_argument("--vocab", dest="vocab", nargs="*", help="File(s) containing words to output embeddings for")
parser.add_argument("--output", required=True, dest="output", default="word_and_morpho_embeds.txt", help="Output location")
parser.add_argument("--lowercase-backoff", dest="lowercase_backoff", action="store_true", help="Use lowercased segmentation if not available for capitalized word")
parser.add_argument("--in-vocab-only", dest="in_vocab_only", action="store_true", help="Only output an embedding if it is in-vocab")
parser.add_argument("--average-unk", dest="average_unk", action="store_true", help="Use average embeddings instead of model-given UNK")
parser.add_argument("--w2v", dest="w2v", action="store_true", help="Model is W2V, not Polyglot")
options = parser.parse_args()

# read in embeddings
if options.w2v:
    words, embs = read_text_embs(options.vectors)
    unk_sym = W2V_UNK
else:
    words, embs = read_pickle_embs(options.vectors)
    unk_sym = POLYGLOT_UNK
print("Total in Embeddings vocabulary: {}".format(len(words)))
word_to_ix = {w : i for (i,w) in enumerate(words)}
if options.average_unk:
    unk_emb = average(embs)
else:
    unk_emb = embs[word_to_ix[unk_sym]]

# Read in the output vocab
if options.vocab is not None:
    output_words = set()
    for filename in options.vocab:
        with codecs.open(filename, "r", "utf-8") as f:
            output_words.update(line.strip() for line in f)
else:
    output_words = set(words)

# intersect vocab and embeddings into output file
word_set = set(words)
with codecs.open(options.output, "w", "utf-8") as outfile:
    in_vocab = 0
    total = len(output_words)
    for orig_word in output_words:
        if orig_word not in word_set and options.lowercase_backoff:
            word = orig_word.lower()
        else:
            word = orig_word
        if word in word_set:
            embed = embs[word_to_ix[word]]
            output_word_vector(orig_word, embed, outfile)
            in_vocab += 1
        elif options.in_vocab_only: continue
        else:
            embed = unk_emb
            output_word_vector(orig_word, embed, outfile)
    print("Total Number of output words:", total)
    print("Total in Training Vocabulary:", in_vocab)
    print("Percentage in-vocab:", in_vocab / total)
