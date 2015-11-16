from collections import defaultdict
from nltk.tree import Tree

class CKYParser:
 
    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        self.grammar = grammar
        self.non_terminals = {x.lhs() for x in grammar.productions()}
        print(self.non_terminals)
 
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
                if len(X_xi) is 0:
                    #pi[(i, i)][str(X)] = 0.0
                    pass
                else:
                    pi[(i, i)][str(X)] = X_xi[0].prob()    
                    bp[(i, i)][str(X)] = (X_xi[0], i)
        #for k in pi:
        #    print(k, pi[k]) 

        for l in range(1, n):
            for i in range(1, n - l + 1):
            	j = i + l
            	for X in self.non_terminals:
                    # rules that have X as lhs
                    X_rules = [rule for rule in self.grammar.productions() if rule.lhs() == X and len(rule.rhs()) is 2]
                    if len(X_rules) is 0:
                    	continue
                    candidates = [(rule.logprob() + pi[(i, s)][str(rule.rhs()[0])] + pi[(s + 1, j)][str(rule.rhs()[1])], rule, s) for rule in X_rules for s in range(i, j)]
                    top = max(candidates, key = lambda x: x[0])
                    pi[(i, j)][str(X)] = top[0]
                    bp[(i, j)][str(X)] = top[1:]

        #for k in pi:
        #    pi[k] = {x: pi[k][x] for x in pi[k] if pi[k][x] > 0.0}
     
        def populate_tree(i, j, lhs):
        	print(i, j, lhs, bp[(i, j)][lhs])
        	if i != j:
        		entry = bp[(i, j)][lhs]
        		return Tree(lhs, [populate_tree(i, entry[1], str(entry[0].rhs()[0])), populate_tree(entry[1] + 1, j, str(entry[0].rhs()[1]))])
        	else:
        		return Tree(lhs, [str([bp[(i, j)][lhs][0].rhs()[0]])])
        	    	
        for k in pi:
        	print(k, pi[k]) 
        populate_tree(1, n, 'S').pretty_print()

        return pi[i, j], populate_tree(1, n, 'S')


