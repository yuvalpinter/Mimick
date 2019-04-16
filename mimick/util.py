def wordify(instance, i2c):
    return ''.join([i2c[i] for i in instance.chars])

def charseq(word, c2i):
    chars = []
    for c in word:
        if c not in c2i:
            c2i[c] = len(c2i)
        chars.append(c2i[c])
    return chars

def read_text_embs(files):
    '''
    copied over from parent dir's utils.py
    '''
    word_embs = dict()
    for filename in files:
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                split = line.split()
                if len(split) > 2:
                    word_embs[split[0]] = np.array([float(s) for s in split[1:]])
    return list(zip(*iter(word_embs.items())))

def read_pickle_embs(files):
    '''
    copied over from parent dir's utils.py
    '''
    word_embs = dict()
    for filename in files:
        print(filename)
        words, embs = pickle.load(open(filename, "r"))
        word_embs.update(list(zip(words, embs)))
    return list(zip(*iter(word_embs.items())))
