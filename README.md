# Mimick
Code for [Mimicking Word Embeddings using Subword RNNs](https://arxiv.org/abs/1707.06961) (EMNLP 2017)

I'm adding details to this documentation as I go. When I'm through, this comment will be gone.

## Dependencies
The main dependency for this project is DyNet. Get it [here](http://dynet.readthedocs.io/en/latest/python.html). Their 2.0 version has just been released, and I hope to upgrade this project and models to that version at some point.

## Create Mimick models
The [mimick](mimick) directory contains scripts relevant to the Mimick model: dataset creation, model creation, intrinsic analysis. The [models](mimick/models) directory within contains models trained for all 23 languages mentioned in the paper. If you're using the pre-trained models, you don't need anything else from the [mimick](mimick) directory in order to run the tagging model. If you train new models, please add them here via pull request!

## Tag parts-of-speech and morphosyntactic attributes using trained models
The root directory of this repository contains the code required to perform extrinsic analysis on Universal Dependencies data. Vocabulary files are supplied in the [vocabs](vocabs) directory.

The entry point is [model.py](model.py), which can use tagging datasets created using the [make_dataset.py](make_dataset.py) script.
Note that `model.py` accepts pre-trained Word Embedding models via **text files** with no header. For Mimick models, this exact format is output into the path in [mimick/model.py](mimick/model.py) script's `--output` argument. For Word2Vec, FastText, or Polyglot models, one can create such a file using the [scripts/output_word_vectors.py](scripts/output_word_vectors.py) script that accepts a model (.pkl or .bin) and the desired output vocabulary (.txt).

## CNN Experiment (October 2017)
References:
- Zhang et al., 2015. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
- dos Santos and Zadrozny, 2014. [Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf). Applied to POS tagging.
- Shen et al., CIKM 2014. [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](https://dl.acm.org/citation.cfm?id=2661935). Extracted from word n-grams, applied to IR.
- Kanaris et al., 2007. [Words Versus Character N-Grams For Anti-Spam Filtering](http://www.worldscientific.com/doi/abs/10.1142/S0218213007003692).