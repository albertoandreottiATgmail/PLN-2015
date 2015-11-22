import nltk
from nltk import grammar
from nltk import Nonterminal as N
from parsing.cky_parser import CKYParser
from parsing.util import unlexicalize
from copy import deepcopy


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence', markov_window = None):
        """
        parsed_sents -- list of training trees.
        """
        prods =  []
        count = 0
        for t in parsed_sents:
            count += 1   
            t2 = deepcopy(t)
            unlexicalize(t2)
            t2.collapse_unary(collapsePOS = True)
            t2.chomsky_normal_form(factor='right', horzMarkov = markov_window)
            [prods.append(p) for p in t2.productions()]

        self._grammar = grammar.induce_pcfg(grammar.Nonterminal(start), prods)    
        print('is chomsky?: ', self._grammar.is_chomsky_normal_form())
        print('is binarised?: ', self._grammar.is_binarised())
        self._parser = CKYParser(self._grammar)
        self._parser._is_lexicalized = False
        self._parser._start = str(self._grammar.start())
        print('start symbol::', str(self._grammar.start()))

 
    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        
        return self._parser.parse([x[1] for x in tagged_sent], [x[0] for x in tagged_sent])[1]

