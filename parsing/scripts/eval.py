"""Evaulate a parser.

Usage:
  eval.py -i <file> [-m <length>] [-n <limit>]
  eval.py -h | --help

Options:
  -i <file>     Parsing model file.
  -m <m>        Parse only sentences of length <= <m>.
  -n <n>        Parse only <n> sentences (useful for profiling).
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader
from datetime import datetime
from parsing.util import spans


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()

if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading model...')
    filename = opts['-i']
    limit = int(opts['-n']) if opts['-n'] is not None else sys.maxsize
    length = int(opts['-m']) if opts['-m'] is not None else sys.maxsize

    f = open(filename, 'rb')
    model = pickle.load(f)
    model._start = 'sentence'
    f.close()

    print('Loading corpus...')
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    parsed_sents = list(corpus.parsed_sents())

    print('Parsing...')
    t1 = datetime.now()
    hits, unlabelled_hits, total_gold, total_model = 0, 0, 0, 0
    
    if opts['-m'] is not None:
        length = int(opts['-m']) 
        parsed_sents = [tree for tree in parsed_sents if len(tree.pos()) <= length]
        n = len(parsed_sents)
    else:
        length = sys.maxsize
        n = len(parsed_sents)
    format_str = '{:3.1f}% ({}/{}) (P={:2.2f}%, R={:2.2f}%, F1={:2.2f}%)'
    progress(format_str.format(0.0, 0, n, 0.0, 0.0, 0.0))
    for i, gold_parsed_sent in enumerate(parsed_sents):

        if i > limit - 1:
            break

        tagged_sent = gold_parsed_sent.pos()

        #gold_parsed_sent.pretty_print()
        #print(gold_parsed_sent)

        if len(tagged_sent) > length:
            continue

        # parse
        model_parsed_sent = model.parse(tagged_sent)

        # compute labelled scores
        gold_spans = spans(gold_parsed_sent, unary=False)
        model_spans = spans(model_parsed_sent, unary=False)
        hits += len(gold_spans & model_spans)

        # unlabelled hits - do not care about the label, ('X', 3, 4) is the same as ('Y', 3, 4)
        unlabelled_hits += len({x[1:] for x in gold_spans} & {x[1:] for x in model_spans})

        total_gold += len(gold_spans)
        total_model += len(model_spans)

        # compute labelled partial results
        prec = float(hits) / total_model * 100
        rec = float(hits) / total_gold * 100
        if prec + rec > 0.0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        # compute labelled partial results
        uprec = float(unlabelled_hits) / total_model * 100
        urec = float(unlabelled_hits) / total_gold * 100
        if uprec + urec > 0.0:
            uf1 = 2 * uprec * urec / (uprec + urec)
        else:
            uf1 = 0.0

        progress(format_str.format(float(i+1) * 100 / n, i + 1, n, prec, rec, f1))

    t2 = datetime.now()
    delta = t2 - t1
    print('')
    print('Parsed {} sentences'.format(n))
    print('Labeled')
    print('  Precision: {:2.2f}% '.format(prec))
    print('  Recall: {:2.2f}% '.format(rec))
    print('  F1: {:2.2f}% '.format(f1))
    print('Unlabelled')
    print('  Precision: {:2.2f}% '.format(uprec))
    print('  Recall: {:2.2f}% '.format(urec))
    print('  F1: {:2.2f}% '.format(uf1))
    print('Total time: {}'.format(delta.seconds))
