from nltk import grammar
from parsing.cky_parser import CKYParser
from parsing.util import unlexicalize, lexicalize
from copy import deepcopy
from nltk.grammar import Nonterminal as N


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence', markov_window=None):
        """
        parsed_sents -- list of training trees.
        """
        prods = []
        for t in parsed_sents:
            t2 = unlexicalize(deepcopy(t))
            t2.chomsky_normal_form(horzMarkov=markov_window)
            t2.collapse_unary(collapsePOS=True, collapseRoot=True)
            [prods.append(p) for p in t2.productions()]

        self._grammar = grammar.induce_pcfg(N(start), prods)
        self._parser = CKYParser(self._grammar)

        print('is chomsky?: ', self._grammar.is_chomsky_normal_form())
        print('is binarised?: ', self._grammar.is_binarised())
        print('start symbol::', str(self._grammar.start()))

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        words = [w for w, t in tagged_sent]
        tags = [t for w, t in tagged_sent]

        # unlex tree
        tree = lexicalize(self._parser.parse(tags)[1], words)
        tree.un_chomsky_normal_form()
        return tree
