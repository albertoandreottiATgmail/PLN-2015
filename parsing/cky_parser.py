from collections import defaultdict
from nltk.tree import Tree
from math import log

log2 = lambda x: log(x, 2.0)

class CKYParser:
 
    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        self.grammar = grammar
        self.non_terminals = {x.lhs() for x in grammar.productions()}
 
    def parse(self, sent):
        """Parse a sequence of terminals.
 
        sent -- the sequence of terminals.
        """
        # initialize
        n = len(sent)
        self._pi = pi = defaultdict(lambda : defaultdict(float))
        self._bp = bp = defaultdict(lambda : defaultdict(float))

        for i in range(1, n + 1):	
            for X in self.non_terminals:
            	# rules that have X as lhs
                X_xi = [rule for rule in self.grammar.productions() if rule.lhs() == X and rule.rhs() == (sent[i - 1], )]
                if len(X_xi) is not 0:
                    pi[(i, i)][str(X)] = X_xi[0].prob()    
                    bp[(i, i)][str(X)] = (X_xi[0], i)

        for l in range(1, n):
            for i in range(1, n - l + 1):
            	j = i + l
            	# TODO: improve this!
            	for X in self.non_terminals:
                    # rules that have X as lhs
                    X_rules = [rule for rule in self.grammar.productions() if rule.lhs() == X and len(rule.rhs()) is 2]
                    if len(X_rules) is 0:
                    	continue
                    candidates = [(rule.prob() * pi[(i, s)][str(rule.rhs()[0])] * pi[(s + 1, j)][str(rule.rhs()[1])], rule, s) for rule in X_rules for s in range(i, j)]

                    top = max(candidates, key = lambda x: x[0])
                    assert(top[0] < 1.0)
                    pi[(i, j)][str(X)] = top[0]
                    bp[(i, j)][str(X)] = top[1:]

        for k in pi:
            print(k, pi[k]) 
     
        def populate_tree(i, j, lhs):
        	if i != j:
        		entry = bp[(i, j)][lhs]
        		return Tree(lhs, [populate_tree(i, entry[1], str(entry[0].rhs()[0])), populate_tree(entry[1] + 1, j, str(entry[0].rhs()[1]))])
        	else:
        		return Tree(lhs, [str(bp[(i, j)][lhs][0].rhs()[0])])
        	    	
        #populate_tree(1, n, 'S').pretty_print()
        return log2(pi[(i, j)]['S']), populate_tree(1, n, 'S')


