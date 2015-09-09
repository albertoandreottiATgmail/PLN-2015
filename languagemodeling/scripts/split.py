"""Split file randomly in train and test sets

Usage:
  split.py -t <t> [-i <file>]
  split.py -h | --help

Options:
  -t <t>        Fraction of the data you want in train. E.g., 0.8.
  -i <file>     Input file.
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

    

    # load the data
    train_fraction = float(opts['-t'])
    file_name = opts['-i'] if opts['-i'] else 'eng-za_web_2013_100K-sentences.txt' 

    test = open(file_name + '_test', "w")
    train = open(file_name + '_train', "w")
    with open(file_name) as corpus:
        for line in corpus:
            train.write(line) if train_fraction > random() else test.write(line)

    test.close()
    train.close()
