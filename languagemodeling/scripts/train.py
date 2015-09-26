"""Train an n-gram model.

Usage:
  train.py -n <n> -o <file> [-a] [-i] [-b]
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -a            Use add one smoothing.
  -i            Use interpolation smoothing.
  -b            Use BackOff smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import gutenberg
from leipzigreader import LeipzigCorpusReader
from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram, BackOffNGram


if __name__ == '__main__':
    opts = docopt(__doc__)
    corpus = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_train')

    # load the data
    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])

    if opts['-i']:
        corpus = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_train_train')
        sents = corpus.sents()
        model = InterpolatedNGram(n, sents)
    elif opts['-b']:
        corpus = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_train_train')
        sents = corpus.sents()
        model = BackOffNGram(n, sents)    
    else:    
        model = AddOneNGram(n, sents) if opts['-a'] else NGram(n, sents)


    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
