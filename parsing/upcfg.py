import nltk
from nltk import grammar
from nltk import Nonterminal as N
from parsing.cky_parser import CKYParser


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        prods =  []
        for t in parsed_sents:
            for p in t.productions():
                if p.is_lexical():
                    prods.append(nltk.grammar.ProbabilisticProduction(N(str(p.lhs())), [str(p.lhs())]))
                else:
                    prods.append(p)

        self._grammar = grammar.induce_pcfg(grammar.Nonterminal(start), prods)
        self._parser = CKYParser(self._grammar)
        self._parser._is_lexicalized = False
 
    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        
        return self._parser.parse([x[1] for x in tagged_sent], [x[0] for x in tagged_sent])[1]
