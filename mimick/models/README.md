Each language's model tarball contains the following:

- `.bin` file with the model parameters, saved using the [model.py](../model.py) script in DyNet 2.0.
- `.c2i` file created separately by [model.py](../model.py) containing the character-to-integer mapping necessary to align embedding parameters in the model with characters in the dataset.

For downstream use, these files should remain in the same directory after untarring.
