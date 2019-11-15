# Mimick
Code for [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010) (EMNLP 2017) and subsequent experiments.

## tl;dr
Given a word embedding dictionary (with vectors from, e.g. FastText or Polyglot or GloVe), Mimick trains a character-level neural net that learns to approximate the embeddings. It can then be applied to infer embeddings in the same space for words that were not available in the original set (i.e. OOVs - Out Of Vocabulary).

## Citation

Please cite our paper if you use this code.

```latex
@inproceedings{pinter2017mimicking,
  title={Mimicking Word Embeddings using Subword RNNs},
  author={Pinter, Yuval and Guthrie, Robert and Eisenstein, Jacob},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={102--112},
  year={2017}
}
```

## Dependencies
The main dependency for this project is DyNet. Get it [here](http://dynet.readthedocs.io/en/latest/python.html).

* As of November 22, 2017, the code complies with Dynet 2.0. You may access the 1.0 version code via the commit log.

## Create Mimick models
The [mimick](mimick) directory contains scripts relevant to the Mimick model: dataset creation, model creation, intrinsic analysis (see readme within). The [models](mimick/models) directory within contains models trained for all 23 languages mentioned in the paper. If you're using the pre-trained models, you don't need anything else from the [mimick](mimick) directory in order to run the [tagging model](#tag-parts-of-speech-and-morphosyntactic-attributes-using-trained-models). If you train new models, please add them here via pull request!

* December 12, 2017 note: pre-trained model are now in DyNet 2.0 format (and employ early-stopping). The 1.0-compatible models are still available [in a subdirectory](mimick/models/dynet_v1).

### CNN Version (November 2017)

As of the November 22 [PR](https://github.com/yuvalpinter/Mimick/pull/2), there is a CNN version of Mimick available for training. It is currently a single-layer convolutional net (conv -> ReLU -> max-k-pool -> fully-connected -> tanh -> fully-connected) that performs the same function as the LSTM version.

## Tag parts-of-speech and morphosyntactic attributes using trained models
The root directory of this repository contains the code required to perform extrinsic analysis on Universal Dependencies data. Vocabulary files are supplied in the [vocabs](vocabs) directory.

The entry point is [model.py](model.py), which can use tagging datasets created using the [make_dataset.py](make_dataset.py) script.
Note that `model.py` accepts pre-trained Word Embedding models via **text files** with no header. For Mimick models, this exact format is output into the path in [mimick/model.py](mimick/model.py) script's `--output` argument. For Word2Vec, FastText, or Polyglot models, one can create such a file using the [scripts/output_word_vectors.py](scripts/output_word_vectors.py) script that accepts a model (`.pkl` or `.bin`) and the desired output vocabulary (`.txt`).
