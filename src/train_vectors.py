"""
v2 of train_embedding
"""
import argparse

import pickle

import gensim
import logging

from util import bcolors
from util import init_training
from util import stringnorm
from util import SentIter

# tmp
#import time


def main():
    # TODO: accept input
    start = time.time()

    dpath = "dat"# TODO: as CLI input
    #dpath = "dat-error"

    init_training()
    print(f"{bcolors.OKGREEN}[INFO] preparing data ...{bcolors.ENDC}")
    sentences = SentIter(dpath)

    print(f"{bcolors.OKGREEN}[INFO] initiating model ...{bcolors.ENDC}")
    mdl = gensim.models.Word2Vec(vector_size=50, window=5, min_count=10, workers=8)# update vector size
    mdl.build_vocab(sentences)
    mdl.train(sentences, total_examples=mdl.corpus_count, epochs=mdl.epochs)# update epochs
    # TODO: store model
    #mdl.save("mdl/word2vec.model")# TODO: make as input path with tag

    print(f"{bcolors.OKGREEN}[INFO] writing vectors to disc ...{bcolors.ENDC}")
    lexicon = list(mdl.wv.index_to_key)

    db = dict()
    for word in lexicon: 
        #db[word] = mdl.wv[word]
        # alternate options
        db[word] = mdl.wv.get_vector(word)
        #db[word] = model.wv.word_vec(word, use_norm=False)
    
    fname = "mdl/vectors_expr8.pcl"# TODO: make as input path with tag
    with open(fname, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"{bcolors.OKGREEN}[INFO] saved {len(lexicon)} vectors to {fname}{bcolors.ENDC}")

    print(f"\n[INFO] runtime {time.time()-start} seconds.")


if __name__=="__main__":
    main()