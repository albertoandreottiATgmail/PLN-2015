import nltk
from nltk import grammar
from nltk import Nonterminal as N
from parsing.cky_parser import CKYParser
from parsing.util import unlexicalize
from copy import deepcopy


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        prods =  []

        for t in parsed_sents:
            if t.height() > 6:
                continue
   
            t2 = deepcopy(t)
            #print(t2.leaves())
            unlexicalize(t2)
            t2.chomsky_normal_form(factor='right', horzMarkov=0)
            t2.collapse_unary(collapsePOS = True)
            for st in t2.subtrees():
                #st.chomsky_normal_form(factor='right', horzMarkov=0)
                st.collapse_unary(collapsePOS = True)
            t2.pretty_print()
            [prods.append(p) for p in t2.productions()]

        self._grammar = grammar.induce_pcfg(grammar.Nonterminal(start), prods)    
        print('is chomsky?: ', self._grammar.is_chomsky_normal_form())
        self._parser = CKYParser(self._grammar)
        self._parser._is_lexicalized = False
        self._parser._start = str(self._grammar.start())
        print('start::', str(self._grammar.start()))
        #for p in self._grammar.productions():
        #    print(p)

 
    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        
        return self._parser.parse([x[1] for x in tagged_sent], [x[0] for x in tagged_sent])[1]
