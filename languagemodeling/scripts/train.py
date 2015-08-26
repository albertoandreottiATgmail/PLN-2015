"""Train an n-gram model.

Usage:
  train.py -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import gutenberg
from leipzigreader import LeipzigCorpusReader
from languagemodeling.ngram import NGram


if __name__ == '__main__':
    opts = docopt(__doc__)
    corpus = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt')

    # load the data
    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])
    model = NGram(n, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
