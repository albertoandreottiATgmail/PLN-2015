"""Train a sequence tagger.

Usage:
  train.py [-m <model>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: base]:
                  base, Baseline
                  hmm:n, MLHMM with order n.
                  memm:n MEMM with order n.

  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM
from tagging.memm import MEMM
from tagging.vector import VectorTagger


models = {
    'base': BaselineTagger, 'hmm': MLHMM, 'memm': MEMM, 'vector:logreg': VectorTagger, 'vector:mlp': VectorTagger
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('./ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    first_chunk = opts['-m'].split(':')[1]
    if opts['-m'].startswith('hmm') or opts['-m'].startswith('memm'):
        model = models[opts['-m'].split(':')[0]](int(first_chunk), sents)
    else:
        model = models[opts['-m']](first_chunk, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
