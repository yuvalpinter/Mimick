# Mimick
Code for Mimicking Word Embeddings using Subword RNNs (EMNLP 2017)

Detailed documentation coming soon. In the meantime, a brief overview of the repository:

- The [mimick](mimick) directory contains scripts relevant to the Mimick model: dataset creation, model creation, intrinsic analysis. The [models](mimick/models) directory within contains models trained for all 23 languages in the paper.

- The main directory contains the code required to perform extrinsic analysis on Universal Dependencies data (vocabulary files are supplied in the [vocabs](vocabs) directory.

The entry point is [model.py](model.py), which can use tagging datasets created using the [make_dataset.py](make_dataset.py) script.
