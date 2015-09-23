"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
  """

from docopt import docopt
import pickle
from random import random
from math import log, pow

from nltk.corpus import gutenberg
from leipzigreader import LeipzigCorpusReader
from languagemodeling.ngram import NGram, AddOneNGram, compute_ngram, compute_1gram



if __name__ == '__main__':

    opts = docopt(__doc__)
    # train the model
    model_file = str(opts['-i'])
    model = pickle.load(open(model_file, "rb"))

    test_set = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_test')
    log2 = lambda x: log(x, 2)
    pow2 = lambda x: pow(2.0, x)
    
    if model.n > 1:
        log_probability, M = compute_ngram(test_set, model)
    else:
        log_probability, M = compute_1gram(test_set, model)    


    cross_entropy = -log_probability / float(M)
    perplexity = pow2(cross_entropy)
    print('Model N: %d' % model.n)
    print('Perplexity: %f, log probability: %f, cross entropy: %f' % (perplexity, log_probability, cross_entropy))



