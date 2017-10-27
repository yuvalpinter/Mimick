'''
Script for training Mimick model to predict OOV word embeddings based on pre-trained embeddings dictionary.
'''
from __future__ import division
from collections import Counter

import collections
import argparse
import random
import cPickle
import logging
import progressbar
import os
import math
import datetime
import codecs
import dynet as dy
import numpy as np

from util import wordify, charseq

__author__ = "Yuval Pinter, 2017"

POLYGLOT_UNK = unicode("<UNK>")
PADDING_CHAR = "<*>"

DEFAULT_CHAR_DIM = 20
DEFAULT_HIDDEN_DIM = 50
DEFAULT_WORD_DIM = 64
DEFAULT_WINDOW_WIDTH = 3
DEFAULT_POOLING_MAXK = 1

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

######################################

class CNNMimick:
    '''
    Implementation details inferred from
    http://dynet.readthedocs.io/en/latest/python_ref.html#convolution-pooling-operations
    '''

    def __init__(self, c2i, num_conv_layers=1, char_dim=DEFAULT_CHAR_DIM, hidden_dim=DEFAULT_HIDDEN_DIM,\
                window_width=DEFAULT_WINDOW_WIDTH, pooling_maxk=DEFAULT_POOLING_MAXK, stride=[1,1],\
                word_embedding_dim=DEFAULT_WORD_DIM, file=None):
        self.c2i = c2i
        self.pooling_maxk = pooling_maxk
        self.stride = stride # TODO change so first is fixed and stride is int param
        self.char_dim = char_dim
        self.hidden_dim = hidden_dim
        self.window_width = window_width
        self.model = dy.Model()
        
        #if file == None:
        ### TODO allow more layers (create list of length num_conv_layers,\
        ### don't forget max-pooling after each in predict_emb)
        self.char_lookup = self.model.add_lookup_parameters((len(c2i), char_dim), name="ce")
        self.conv = self.model.add_parameters((1, window_width, char_dim, hidden_dim), name="conv")
        self.conv_bias = self.model.add_parameters((hidden_dim), name="convb")
        
        self.cnn_to_rep_params = self.model.add_parameters((word_embedding_dim, hidden_dim * pooling_maxk), name="H")
        self.cnn_to_rep_bias = self.model.add_parameters(word_embedding_dim, name="Hb")
        self.mlp_out = self.model.add_parameters((word_embedding_dim, word_embedding_dim), name="O")
        self.mlp_out_bias = self.model.add_parameters(word_embedding_dim, name="Ob")
        
        if file is not None:
            ### TODO current problem - version only supports explicit loading into params, so
            ### dimensionalities all need to be specified in init?
            self.model.populate(file)
 
    def predict_emb(self, chars):
        dy.renew_cg()

        ### TODO find out if this row needs replacement for init (probably not)
        # finit = self.char_fwd_lstm.initial_state()

        H = dy.parameter(self.cnn_to_rep_params)
        Hb = dy.parameter(self.cnn_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        
        conv_param = dy.parameter(self.conv)
        conv_param_bias = dy.parameter(self.conv_bias)

        pad_char = self.c2i[PADDING_CHAR]
        padding_size = self.window_width // 2 # TODO also consider w-stride?
        char_ids = ([pad_char] * padding_size) + chars + ([pad_char] * padding_size)
        if len(chars) < self.pooling_maxk:
            # allow k-max pooling layer output to transform to affine
            char_ids.extend([pad_char] * (self.pooling_maxk - len(chars)))
        
        embeddings = dy.concatenate_cols([self.char_lookup[cid] for cid in char_ids])
        reshaped_embeddings = dy.reshape(dy.transpose(embeddings), (1, len(char_ids), self.char_dim))
        
        ### TODO I might want to change these to is_valid=False, need to think about logic of padding
        ### TODO try with no bias
        conv_out = dy.conv2d_bias(reshaped_embeddings, conv_param, conv_param_bias, self.stride, is_valid=True)
        
        relu_out = dy.rectify(conv_out)
        
        ### pooling when max_k can only be 1, not sure what other differences may be
        #poolingk = [1, len(chars)]
        #pooling_out = dy.maxpooling2d(relu_out, poolingk, self.stride, is_valid=True)
        #pooling_out_flat = dy.reshape(pooling_out, (self.hidden_dim,))
        
        pooling_out = dy.kmax_pooling(relu_out, self.pooling_maxk, d=1) # d = what dimension to max over
        pooling_out_flat = dy.reshape(pooling_out, (self.hidden_dim * self.pooling_maxk,))

        return O * dy.tanh(H * pooling_out_flat + Hb) + Ob

    def loss(self, observation, target_rep):
        return dy.squared_distance(observation, dy.inputVector(target_rep))

    def set_dropout(self, p):
        # TODO see if supported/needed
        pass

    def disable_dropout(self):
        # TODO see if supported/needed
        pass

    def save(self, file_name):
        self.model.save(file_name)
        # character mapping saved separately
        cPickle.dump(self.c2i, open(file_name[:-4] + '.c2i', 'w'))

    @property
    def model(self):
        return self.model
        
######################################

class LSTMMimick:

    def __init__(self, c2i, num_lstm_layers=-1,\
                char_dim=-1, hidden_dim=-1, word_embedding_dim=-1, file=None):
        self.c2i = c2i
        self.model = dy.Model()
        if file == None:
            # Char LSTM Parameters
            self.char_lookup = self.model.add_lookup_parameters((len(c2i), char_dim))
            self.char_fwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.model)
            self.char_bwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.model)

            # Post-LSTM Parameters
            self.lstm_to_rep_params = self.model.add_parameters((word_embedding_dim, hidden_dim * 2))
            self.lstm_to_rep_bias = self.model.add_parameters(word_embedding_dim)
            self.mlp_out = self.model.add_parameters((word_embedding_dim, word_embedding_dim))
            self.mlp_out_bias = self.model.add_parameters(word_embedding_dim)
        else:
            ### TODO implement loading from dynet v. 2.0 style save
            # read from saved file
            model_members = iter(self.model.load(file))
            self.char_lookup = model_members.next()
            self.char_fwd_lstm = model_members.next()
            self.char_bwd_lstm = model_members.next()
            self.lstm_to_rep_params = model_members.next()
            self.lstm_to_rep_bias = model_members.next()
            self.mlp_out = model_members.next()
            self.mlp_out_bias = model_members.next()

    def predict_emb(self, chars):
        dy.renew_cg()

        finit = self.char_fwd_lstm.initial_state()
        binit = self.char_bwd_lstm.initial_state()

        H = dy.parameter(self.lstm_to_rep_params)
        Hb = dy.parameter(self.lstm_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)

        pad_char = self.c2i[PADDING_CHAR]
        char_ids = [pad_char] + chars + [pad_char]
        embeddings = [self.char_lookup[cid] for cid in char_ids]

        bi_fwd_out = finit.transduce(embeddings)
        bi_bwd_out = binit.transduce(reversed(embeddings))

        rep = dy.concatenate([bi_fwd_out[-1], bi_bwd_out[-1]])

        return O * dy.tanh(H * rep + Hb) + Ob

    def loss(self, observation, target_rep):
        return dy.squared_distance(observation, dy.inputVector(target_rep))

    def set_dropout(self, p):
        self.char_fwd_lstm.set_dropout(p)
        self.char_bwd_lstm.set_dropout(p)

    def disable_dropout(self):
        self.char_fwd_lstm.disable_dropout()
        self.char_bwd_lstm.disable_dropout()

    def save(self, file_name):
        ### TODO implement dynet 2.0 version
        self.model.save(file_name)

        # character mapping saved separately
        cPickle.dump(self.c2i, open(file_name[:-4] + '.c2i', 'w'))
        
    def old_save(self, file_name):
        members_to_save = []
        members_to_save.append(self.char_lookup)
        members_to_save.append(self.char_fwd_lstm)
        members_to_save.append(self.char_bwd_lstm)
        members_to_save.append(self.lstm_to_rep_params)
        members_to_save.append(self.lstm_to_rep_bias)
        members_to_save.append(self.mlp_out)
        members_to_save.append(self.mlp_out_bias)
        self.model.save(file_name, members_to_save)

        # character mapping saved separately
        cPickle.dump(self.c2i, open(file_name[:-4] + '.c2i', 'w'))

    @property
    def model(self):
        return self.model

def dist(instance, vec):
    we = instance.word_emb
    if options.cosine:
        return 1.0 - (we.dot(vec) / (np.linalg.norm(we) * np.linalg.norm(vec)))
    return np.linalg.norm(we - vec)

if __name__ == "__main__":

    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
    parser.add_argument("--vocab", required=True, dest="vocab", help="total vocab to output")
    parser.add_argument("--output", dest="output", help="file with all embeddings")
    parser.add_argument("--model-out", dest="model_out", help="file with model parameters")
    parser.add_argument("--lang", dest="lang", default="en", help="language")
    parser.add_argument("--char-dim", default=DEFAULT_CHAR_DIM, dest="char_dim", help="dimension for character embeddings (default = 20)")
    parser.add_argument("--hidden-dim", default=DEFAULT_HIDDEN_DIM, dest="hidden_dim", help="dimension for LSTM layers (default = 50)")
    ### LSTM ###
    parser.add_argument("--num-lstm-layers", default=1, dest="num_lstm_layers", help="Number of LSTM layers (default = 1)")
    ### CNN ###
    parser.add_argument("--use_cnn", dest="cnn", action="store_true", help="if toggled, train CNN and not LSTM")
    parser.add_argument("--num-conv-layers", type=int, default=1, dest="num_conv_layers", help="Number of CNN layers (default = 1)")
    parser.add_argument("--window-width", type=int, default=3, dest="window_width", help="Width of CNN layers (default = 3)")
    parser.add_argument("--pooling-maxk", type=int, default=1, dest="pooling_maxk", help="K for K-max pooling (default = 1)")
    parser.add_argument("--stride", default=[1,1], dest="stride", help="Stride for CNN layers (default = [1,1])")
    ### END ###
    parser.add_argument("--all-from-mimick", dest="all_from_mimick", action="store_true", help="if toggled, vectors in original training set are overriden by Mimick-generated vectors")
    parser.add_argument("--normalized-targets", dest="normalized_targets", action="store_true", help="if toggled, train on normalized vectors from set")
    parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="amount of dropout to apply to LSTM part of graph")
    parser.add_argument("--num-epochs", default=10, dest="num_epochs", type=int, help="Number of full passes through training set (default = 10)")
    parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
    parser.add_argument("--cosine", dest="cosine", action="store_true", help="Use cosine as diff measure")
    parser.add_argument("--dynet-mem", help="Ignore this outside argument")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
    options = parser.parse_args()

    # Set up logging
    log_dir = "embedding_train_mimick-{}-{}{}".format(datetime.datetime.now().strftime('%y%m%d%H%M%S'),\
                                                      options.lang, '-DEBUG' if options.debug else '')
    os.mkdir(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir + '/log.txt', 'w', 'utf-8')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    root_logger.info("Training dataset: {}".format(options.dataset))
    root_logger.info("Output vocabulary: {}".format(options.vocab))
    root_logger.info("Vectors output location: {}".format(options.output))
    root_logger.info("Model output location: {}\n".format(options.model_out))

    # Load training set
    dataset = cPickle.load(open(options.dataset, "r"))
    c2i = dataset["c2i"]
    i2c = { i: c for c, i in c2i.items() } # inverse map
    training_instances = dataset["training_instances"]
    test_instances = dataset["test_instances"]
    populate_test_insts_from_vocab = len(test_instances) == 0
    emb_dim = len(training_instances[0].word_emb)

    # Load words to write
    vocab_words = {}
    if populate_test_insts_from_vocab:
        train_words = [wordify(w, i2c) for w in training_instances]
    with codecs.open(options.vocab, "r", "utf-8") as vocab_file:
        for vw in vocab_file.readlines():
            vw = vw.strip()
            vocab_words[vw] = np.zeros(emb_dim)
            if populate_test_insts_from_vocab and vw not in train_words:
                test_instances.append(Instance(charseq(vw, c2i), np.zeros(emb_dim)))
    
    if populate_test_insts_from_vocab:
        # might need to update i2c
        i2c = { i: c for c, i in c2i.items() }

    if not options.cnn:
        model = LSTMMimick(c2i, options.num_lstm_layers, options.char_dim, options.hidden_dim, emb_dim)
    else:
        model = CNNMimick(c2i, options.num_conv_layers, options.char_dim, options.hidden_dim,\
                options.window_width, options.pooling_maxk, options.stride, emb_dim)
    trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
    root_logger.info("Training Algorithm: {}".format(type(trainer)))

    root_logger.info("Number training instances: {}".format(len(training_instances)))

    # Create dev set
    random.shuffle(training_instances)
    dev_cutoff = int(99 * len(training_instances) / 100)
    dev_instances = training_instances[dev_cutoff:]
    training_instances = training_instances[:dev_cutoff]

    if options.debug:
        train_instances = training_instances[:int(len(training_instances)/20)]
        dev_instances = dev_instances[:int(len(dev_instances)/20)]
    else:
        train_instances = training_instances

    if options.normalized_targets:
        train_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in train_instances]
        dev_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in dev_instances]

    epcs = int(options.num_epochs)
    pretrained_vec_norms = 0.0
    inferred_vec_norms = 0.0

    # Shuffle set, divide into cross-folds each epoch
    for epoch in xrange(epcs):
        bar = progressbar.ProgressBar()

        train_loss = 0.0
        train_correct = Counter()
        train_total = Counter()

        if options.dropout > 0:
            model.set_dropout(options.dropout)

        for instance in bar(train_instances):
            if len(instance.chars) <= 0: continue
            obs_emb = model.predict_emb(instance.chars)
            loss_expr = model.loss(obs_emb, instance.word_emb)
            loss = loss_expr.scalar_value()

            # Bail if loss is NaN
            if math.isnan(loss):
                assert False, "NaN occured"

            train_loss += loss

            # Do backward pass and update parameters
            loss_expr.backward()
            trainer.update()

            if epoch == epcs - 1:
                word = wordify(instance, i2c)
                if word in vocab_words:
                    pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                    if options.all_from_mimick:
                        vocab_words[word] = np.array(obs_emb.value())
                        inferred_vec_norms += np.linalg.norm(vocab_words[word])
                    else: # log vocab embeddings
                        vocab_words[word] = instance.word_emb

        root_logger.info("\n")
        root_logger.info("Epoch {} complete".format(epoch + 1))
        # here used to be a learning rate update, no longer supported in dynet 2.0
        print trainer.status()

        # Evaluate dev data
        model.disable_dropout()
        dev_loss = 0.0
        dev_correct = Counter()
        dev_total = Counter()

        bar = progressbar.ProgressBar()
        for instance in bar(dev_instances):
            if len(instance.chars) <= 0: continue
            obs_emb = model.predict_emb(instance.chars)
            dev_loss += model.loss(obs_emb, instance.word_emb).scalar_value()

            if epoch == epcs - 1:
                word = wordify(instance, i2c)
                if word in vocab_words:
                    pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                    if options.all_from_mimick:
                        vocab_words[word] = np.array(obs_emb.value())
                        inferred_vec_norms += np.linalg.norm(vocab_words[word])
                    else: # log vocab embeddings
                        vocab_words[word] = instance.word_emb

        root_logger.info("Train Loss: {}".format(train_loss))
        root_logger.info("Dev Loss: {}".format(dev_loss))

    root_logger.info("\n")
    root_logger.info("Average norm for pre-trained in vocab: {}".format(pretrained_vec_norms / len(vocab_words)))

    # Infer for test set
    showcase_size = 5
    top_to_show = 10
    showcase = [] # sample for similarity sanity check
    for idx, instance in enumerate(test_instances):
        word = wordify(instance, i2c)
        obs_emb = model.predict_emb(instance.chars)
        vocab_words[word] = np.array(obs_emb.value())
        inferred_vec_norms += np.linalg.norm(vocab_words[word])

        if options.debug:
            # reservoir sampling
            if idx < showcase_size:
                showcase.append(word)
            else:
                rand = random.randint(0,idx-1)
                if rand < showcase_size:
                    showcase[rand] = word

    root_logger.info("Average norm for trained: {}".format(inferred_vec_norms / len(test_instances)))

    if options.debug:
        similar_words = {}
        for w in showcase:
            vec = vocab_words[w]
            top_k = [(wordify(instance, i2c),d) for instance,d in sorted([(inst, dist(inst, vec)) for inst in training_instances], key=lambda x: x[1])[:top_to_show]]
            print w, [(i,d) for i,d in top_k]
            similar_words[w] = top_k


    # write all
    if options.output is not None:
        with codecs.open(options.output, "w", "utf-8") as writer:
            writer.write("{} {}\n".format(len(vocab_words), emb_dim))
            for vw, emb in vocab_words.iteritems():
                writer.write(vw + " ")
                for i in emb:
                    writer.write("{:.6f} ".format(i))
                writer.write("\n")

    # save model
    if options.model_out is not None:
        model.save(options.model_out)
