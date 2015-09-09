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

from nltk.corpus import gutenberg
from leipzigreader import LeipzigCorpusReader
from languagemodeling.ngram import NGram, AddOneNGram


if __name__ == '__main__':

    opts = docopt(__doc__)
    # train the model
    model_file = str(opts['-i'])
    model = pickle.load(open(model_file, "rb"))

    test_set = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_test')
    words = test_set.words()
    prev_tokens = words[ : model.n - 1]
    perplexity = model.cond_prob(prev_tokens, words[model.n - 1])

    for word in test_set[model.n + 1 : ]:
        perplexity *= model.prob(word, prev_tokens)
        prev_tokens = prev_tokens[1:] + [word]

    perplexity = 1 / perplexity

