"""
Reads in CONLL files to make the dataset
Output a textual vocabulary file, and a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2is: Dict mapping attribute types (POS / morpho) to dicts from tags to indices
c2i: Dict mapping characters to indices
"""

from _collections import defaultdict
import codecs
import argparse
import cPickle
import collections
from utils import split_tagstring

__author__ = "Yuval Pinter and Robert Guthrie, 2017"

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

def read_file(filename, w2i, t2is, c2i, options):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars
    as it sees them.
    """

    # populate mandatory t2i tables
    if POS_KEY not in t2is:
        t2is[POS_KEY] = {}

    # build dataset
    instances = []
    vocab_counter = collections.Counter()
    with codecs.open(filename, "r", "utf-8") as f:

        # running sentence buffers (lines are tokens)
        sentence = []
        tags = defaultdict(list)

        # main file reading loop
        for i, line in enumerate(f):

            # discard comments
            if line.startswith("#"):
                continue

            # parse sentence end
            elif line.isspace():

                # pad tag lists to sentence end
                slen = len(sentence)
                for seq in tags.values():
                    if len(seq) < slen:
                        seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG

                # add sentence to dataset
                instances.append(Instance(sentence, tags))
                sentence = []
                tags = defaultdict(list)

            else:

                # parse token information in line
                data = line.split("\t")
                if '-' in data[0]: # Some UD languages have contractions on a separate line, we don't want to include them also
                    continue
                idx = int(data[0])
                word = data[1]
                postag = data[3] if options.ud_tags else data[4]
                morphotags = {} if options.no_morphotags else split_tagstring(data[5], uni_key=False)

                # ensure counts and dictionary population
                vocab_counter[word] += 1
                if word not in w2i:
                    w2i[word] = len(w2i)
                pt2i = t2is[POS_KEY]
                if postag not in pt2i:
                    pt2i[postag] = len(pt2i)
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                for key, val in morphotags.items():
                    if key not in t2is:
                        t2is[key] = {NONE_TAG:0}
                    mt2i = t2is[key]
                    if val not in mt2i:
                        mt2i[val] = len(mt2i)

                # add data to sentence buffer
                sentence.append(w2i[word])
                tags[POS_KEY].append(t2is[POS_KEY][postag])
                for k,v in morphotags.items():
                    mtags = tags[k]
                    # pad backwards to latest seen
                    missing_tags = idx - len(mtags) - 1
                    mtags.extend([0] * missing_tags) # 0 guaranteed above to represent NONE_TAG
                    mtags.append(t2is[k][v])

    return instances, vocab_counter

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
    parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
    parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
    parser.add_argument("--ud-tags", dest="ud_tags", action="store_true", help="Extract UD tags instead of original tags")
    parser.add_argument("--no-morphotags", dest="no_morphotags", action="store_true", help="Don't add morphosyntactic tags to dataset")
    parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
    parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                        the train/dev/test data to use in outputting embeddings")
    options = parser.parse_args()

    w2i = {} # mapping from word to index
    t2is = {} # mapping from attribute name to mapping from tag to index
    c2i = {} # mapping from character to index, for char-RNN concatenations
    output = {}

    # read data from UD files
    output["training_instances"], output["training_vocab"] = read_file(options.training_data, w2i, t2is, c2i, options)
    output["dev_instances"], output["dev_vocab"] = read_file(options.dev_data, w2i, t2is, c2i, options)
    output["test_instances"], output["test_vocab"] = read_file(options.test_data, w2i, t2is, c2i, options)

    # Add special tokens / tags / chars to dicts
    w2i[UNK_TAG] = len(w2i)
    for t2i in t2is.values():
        t2i[START_TAG] = len(t2i)
        t2i[END_TAG] = len(t2i)
    c2i[PADDING_CHAR] = len(c2i)

    output["w2i"] = w2i
    output["t2is"] = t2is
    output["c2i"] = c2i

    # write outputs to files
    with open(options.output, "w") as outfile:
        cPickle.dump(output, outfile)
    with codecs.open(options.vocab_file, "w", "utf-8") as vocabfile:
        for word in w2i.keys():
            vocabfile.write(word + "\n")
