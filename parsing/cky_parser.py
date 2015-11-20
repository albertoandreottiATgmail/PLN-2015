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
        self._is_lexicalized = True
        self._start = 'S'
 
    def parse(self, sent, lex=[]):
        """Parse a sequence of terminals.
 
        sent -- the sequence of terminals.
        """
        # initialize
        n = len(sent)
        self._pi = pi = defaultdict(lambda : defaultdict(float))
        self._bp = bp = defaultdict(lambda : defaultdict(float))

        for i in range(1, n + 1):	
            for rule in self.grammar.productions():
            	# rules that have X as lhs
                #X_xi = [rule for rule in self.grammar.productions() if rule.lhs() == X and rule.rhs() == (sent[i - 1], )]
                #print('X_xi:',i,X_xi)
                
                if len(rule.rhs()) == 1 and rule.rhs() == (sent[i - 1], ):
                    pi[(i, i)][str(rule.lhs())] = rule.logprob()
                    bp[(i, i)][str(rule.lhs())] = (rule, i)

        for l in range(1, n):
            for i in range(1, n - l + 1):
            	j = i + l
            	for X_rule in self.grammar.productions():
                    if len(X_rule.rhs()) < 2:
                        continue
                    X = X_rule.lhs()

                    candidates = []
                    for s in range(i, j):
                        if str(X_rule.rhs()[0]) in pi[(i, s)] and str(X_rule.rhs()[1]) in pi[(s + 1, j)]:
                            candidates.append((X_rule.logprob() + pi[(i, s)][str(X_rule.rhs()[0])] + pi[(s + 1, j)][str(X_rule.rhs()[1])], s))
                    if len(candidates) == 0:
                        continue

                    top = max(candidates, key = lambda x: x[0])

                    if str(X) in pi[(i, j)]:
                        if top[0] > pi[(i, j)][str(X)]:
                            pi[(i, j)][str(X)] = top[0]
                            bp[(i, j)][str(X)] = (X_rule, top[1])
                    else:
                        pi[(i, j)][str(X)] = top[0]
                        bp[(i, j)][str(X)] = (X_rule, top[1]) 

        def populate_tree(i, j, lhs):
            if i != j:
                entry = bp[(i, j)][lhs]
                return Tree(lhs, [populate_tree(i, entry[1], str(entry[0].rhs()[0])), populate_tree(entry[1] + 1, j, str(entry[0].rhs()[1]))])
            else:
                # use i, j for fetching the word from the sentence in the upcfg case.
                if self._is_lexicalized:
                    return Tree(lhs, [str(bp[(i, j)][lhs][0].rhs()[0])])
                else:
                    return Tree(lhs, [lex[i - 1]])

        if self._start in pi[(1, n)]:
            return pi[(1, n)][self._start], populate_tree(1, n, self._start)
        else:
            return 0.0, Tree(self._start, [Tree(tag, [word]) for tag, word,  in zip(sent, lex)])    


