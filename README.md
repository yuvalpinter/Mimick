# Mimick
Code for [Mimicking Word Embeddings using Subword RNNs](https://arxiv.org/abs/1707.06961) (EMNLP 2017) and subsequent experiments.

I'm adding details to this documentation as I go. When I'm through, this comment will be gone.

## tl;dr
Given a word embedding dictionary (with vectors from, e.g. FastText or Polyglot or GloVe), Mimick trains a character-level neural net that learns to approximate the embeddings. It can then be applied to infer embeddings in the same space for words that were not available in the original set (i.e. OOVs - Out Of Vocabulary).

## Dependencies
The main dependency for this project is DyNet. Get it [here](http://dynet.readthedocs.io/en/latest/python.html). As of November 22, 2017, the code complies with Dynet 2.0. You may access the 1.0 version code via the commit log.

## Create Mimick models
The [mimick](mimick) directory contains scripts relevant to the Mimick model: dataset creation, model creation, intrinsic analysis. The [models](mimick/models) directory within contains models trained for all 23 languages mentioned in the paper. If you're using the pre-trained models, you don't need anything else from the [mimick](mimick) directory in order to run the tagging model. If you train new models, please add them here via pull request!

* November 22, 2017 note: the pre-trained models were saved in DyNet 1.0 format. It is still possible to load them using the `old_load()` function in `mimick/model.py`, but it could be somewhat of a pain. Another option is to use DyNet 1.0 and an old version of this codebase.
I will be re-training and saving models in the new format, but don't expect it too soon.

### CNN Version (November 2017)

As of the November 22 [PR](https://github.com/yuvalpinter/Mimick/pull/2), there is a CNN version of Mimick available for training. It is currently a single-layer convolutional net (conv -> max-k-pool -> fully-connected -> tanh -> fully-connected) that performs the same function as the LSTM version. What is it good for? Wait for the paper 😎

## Tag parts-of-speech and morphosyntactic attributes using trained models
The root directory of this repository contains the code required to perform extrinsic analysis on Universal Dependencies data. Vocabulary files are supplied in the [vocabs](vocabs) directory.

The entry point is [model.py](model.py), which can use tagging datasets created using the [make_dataset.py](make_dataset.py) script.
Note that `model.py` accepts pre-trained Word Embedding models via **text files** with no header. For Mimick models, this exact format is output into the path in [mimick/model.py](mimick/model.py) script's `--output` argument. For Word2Vec, FastText, or Polyglot models, one can create such a file using the [scripts/output_word_vectors.py](scripts/output_word_vectors.py) script that accepts a model (.pkl or .bin) and the desired output vocabulary (.txt).
