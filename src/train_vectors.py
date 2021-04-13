"""
Embedding pipeline for Grundtvig Centeret
"""
import argparse
import os
import re
import pickle
import spacy
import nltk
import gensim
import logging

# global
def init_training():
    logging.basicConfig(
        format="%(asctime)s : %(levelname)  s : %(message)s",
        level=logging.INFO
        )
    global nlp
    nlp = spacy.load("da_core_news_sm")

def stringnorm(s, lemmatize=False, casefold=False, rmpat=[]):
    if casefold:
        s = s.lower()
    if rmpat:
        for pat in rmpat:
            s = re.sub(pat, " ", s)
        s = re.sub(" +", " ", s)
    doc = nlp(s)
    if lemmatize:
        tokens = [token.lemma_ for token in doc]
    else:
        tokens = [str(token) for token in doc]
    
    return tokens

class SentIter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in sorted(os.listdir(self.dirname)[:3]):
            with open(os.path.join(self.dirname, fname), "r") as f:
                text = f.read()
            sentences = list()
            for sentence in nltk.sent_tokenize(text):
                tokens = stringnorm(sentence, lemmatize=True, casefold=True, rmpat=[r"\d+", r"\W+"])
                tokens = [token for token in tokens if len(token) > 1]
                
                if len(tokens) > 4:
                    yield tokens

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    # TODO: accept input
    dpath = "dat"

    init_training()
    print(f"{bcolors.OKGREEN}[INFO] preparing data ...{bcolors.ENDC}")
    sentences = SentIter(dpath)

    print(f"{bcolors.OKGREEN}[INFO] initiating model ...{bcolors.ENDC}")
    mdl = gensim.models.Word2Vec( vector_size=100, window=5, min_count=1, workers=4)# update vector size
    mdl.build_vocab(sentences)
    mdl.train(sentences, total_examples=mdl.corpus_count, epochs=mdl.epochs)# update epochs

    print(f"{bcolors.OKGREEN}[INFO] writing vectors to disc ...{bcolors.ENDC}")
    lexicon = list(mdl.wv.index_to_key)
    word_vectors = mdl.wv
    db = dict()
    for word in lexicon: db[word] = mdl.wv[word]
    
    fname = "mdl/vectors.pcl"# TODO: make as input path with tag
    with open(fname, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{bcolors.OKGREEN}[INFO] saved {len(lexicon)} vectors to {fname}{bcolors.ENDC}")


if __name__=="__main__":
    main()