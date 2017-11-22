#!/usr/bin/env python
'''
An interactive tool for querying a Mimick model for nearest vectors of OOV words.
'''
import sys
import cPickle as pickle
import numpy as np
import collections
import argparse
from model import LSTMMimick, CNNMimick, DEFAULT_CHAR_DIM, DEFAULT_HIDDEN_DIM, DEFAULT_LSTM_LAYERS
from make_dataset import read_text_embs

__author__ = "Yuval Pinter, 2017"

QUITTING_WORDS = ['q', 'Q', 'quit', 'exit']
Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

def dist(v1, v2):
    return 1.0 - (v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

### TODO reconciliate pkl format with gensim and use its built-in KeyedVectors.similar_by_vector() ###

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimick", required=True, dest="mimick", help="Mimick model file")
    parser.add_argument("--use-cnn", action="store_true", dest="use_cnn", help="Use CNN model")
    parser.add_argument("--c2i", required=True, dest="c2i", help="Mimick char-to-integer mapping file")
    parser.add_argument("--vectors", required=True, dest="vectors", help="Pickle file with reference word vectors")
    parser.add_argument("--w2v-format", dest="w2v_format", action="store_true", help="Vector file is in textual w2v format")
    #parser.add_argument("--ktop", dest="ktop", default=10, help="Number of top neighbors to present (optional)")
    parser.add_argument("--char-dim", type=int, default=DEFAULT_CHAR_DIM, dest="char_dim", help="dimension for character embeddings (default = {})".format(DEFAULT_CHAR_DIM))
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, dest="hidden_dim", help="dimension for LSTM layers (default = {})".format(DEFAULT_HIDDEN_DIM))
    parser.add_argument("--num-lstm-layers", type=int, default=DEFAULT_LSTM_LAYERS, dest="num_lstm_layers", help="Number of LSTM layers (default = {})".format(DEFAULT_LSTM_LAYERS))
    opts = parser.parse_args()

    # load vocab
    if opts.w2v_format:
        voc_words, voc_vecs = read_text_embs(opts.vectors)
    else:
        voc_words, voc_vecs = pickle.load(open(opts.vectors))

    we_dim = len(voc_vecs[0])
    
    # load model
    c2i = pickle.load(open(opts.c2i))
    if opts.use_cnn:
        ### TODO add optional params to opts for compliance with file dimensions (see CNNMimick.__init__)
        mimick = CNNMimick(c2i, file=opts.mimick, word_embedding_dim=we_dim)
    else:
        mimick = LSTMMimick(c2i, num_lstm_layers=opts.num_lstm_layers, char_dim=opts.char_dim,\
                            hidden_dim=opts.hidden_dim,\
                            word_embedding_dim=we_dim, file=opts.mimick)

    # prompt
    while True:
        sys.stdout.write('> ')
        next_word = sys.stdin.readline().strip()
        if next_word in QUITTING_WORDS:
            exit()

        word_chars = [c2i[c] for c in next_word]
        pred_vec = mimick.predict_emb(word_chars).value()
        top_k = sorted([(iv, dist(iv_vec, pred_vec)) for iv,iv_vec in zip(voc_words, voc_vecs)], key=lambda x: x[1])[:10]
        print '\n'.join(['{}:\t{:.3f}'.format(near[0], 1.0 - near[1]) for near in top_k])
