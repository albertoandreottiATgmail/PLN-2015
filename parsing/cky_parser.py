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
        self._pi = pi = defaultdict(lambda: defaultdict(float))
        self._bp = bp = defaultdict(lambda: defaultdict(float))

        for i in range(1, n + 1):
            for rule in self.grammar.productions():
                if len(rule.rhs()) == 1 and rule.rhs() == (sent[i - 1], ):
                    rule_lhs = str(rule.lhs())
                    pi[(i, i)][rule_lhs] = rule.logprob()
                    bp[(i, i)][rule_lhs] = (rule, i)

        prods = defaultdict(list)
        [prods[(str(prod.rhs()[0]), str(prod.rhs()[1]))].append(prod) for prod in self.grammar.productions() if len(prod.rhs()) == 2]
        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                from itertools import product
                for s in range(i, j):
                    for Y, Z in product(pi[(i, s)].keys(), pi[(s + 1, j)].keys()):
                        if (Y, Z) in prods:
                            for prod in prods[(Y, Z)]:
                                candidate = (prod.logprob() + pi[(i, s)][Y] + pi[(s + 1, j)][Z], s, prod)
                                X = str(candidate[2].lhs())
                                if X in pi[(i, j)]:
                                    if candidate[0] > pi[(i, j)][X]:
                                        pi[(i, j)][X] = candidate[0]
                                        bp[(i, j)][X] = (candidate[2], candidate[1])
                                else:
                                    pi[(i, j)][X] = candidate[0]
                                    bp[(i, j)][X] = (candidate[2], candidate[1])

        #for k in pi:
        #    print(k, pi[k])
        def populate_tree(u, v, lhs):
            if u != v:
                entry = bp[(u, v)][lhs]
                return Tree(lhs, [populate_tree(u, entry[1], str(entry[0].rhs()[0])),
                            populate_tree(entry[1] + 1, v, str(entry[0].rhs()[1]))])
            else:
                # use i,j to fetch the word from the sentence in the upcfg case
                if self._is_lexicalized:
                    return Tree(lhs, [str(bp[(u, v)][lhs][0].rhs()[0])])
                else:
                    return Tree(lhs, [lex[u - 1]])

        if self._start in pi[(1, n)]:
            tree = populate_tree(1, n, self._start)
            tree.un_chomsky_normal_form()
            return pi[(1, n)][self._start], tree
        else:
            children = [Tree(tag, [word]) for tag, word, in zip(sent, lex)]
            return 0.0, Tree(self._start, children)
