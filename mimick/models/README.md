Each language's model tarball contains the following:

- `.pkl` file with the model parameters, saved using the [../model.py](model.py) script.
- `.pkl.pym` and `.pkl.pyk` files created alongside it as per DyNet's (1.0) serialization package.
- `.c2i` file created separately by [../model.py](model.py) containing the character-to-integer mapping necessary to align embedding parameters in the model with characters in the dataset.

For downstream use, these files should remain in the same directory after untarring.
