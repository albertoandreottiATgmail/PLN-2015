"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader
from collections import defaultdict

def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()

def default_dict():
    return defaultdict(int)

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = corpus.tagged_sents()
    conf_mat = defaultdict(default_dict)


    # tag
    hits, total, hits_known, total_known = 0, 0, 0, 0

    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        hits_known += sum([hits_sent[i] for i in range(len(hits_sent)) if not model.unknown(word_sent[i])]) 
        total += len(sent)
        total_known += len([w for w in word_sent if not model.unknown(w)])

        for gold, predicted in zip(gold_tag_sent, model_tag_sent):
            conf_mat[gold][predicted] += 1


        assert(total_known > 0)
        acc = float(hits) / total

        progress('{:3.1f}% ({:2.2f}%)'.format(float(i) * 100 / n, acc * 100))

    acc = float(hits) / total
    acc_known = float(hits_known) / total_known
    acc_unknown = float(hits - hits_known) / (total - total_known)


    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
    print('Accuracy known: {:2.2f}%'.format(acc_known * 100))
    print('Accuracy unknown: {:2.2f}%'.format(acc_unknown * 100))
    print('Confusion Matrix:')

    sorted_keys = sorted(conf_mat.keys())
    for row in sorted_keys:
        print (row)
        print([(col, conf_mat[row][col]) for col in sorted_keys])
        



