from collections import defaultdict
from nltk.tree import Tree
from itertools import product


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        self.grammar = grammar
        self._start = str(grammar.start())
        self.single_rules = [rule for rule in self.grammar.productions()
                             if len(rule.rhs()) == 1]
        self.double_rules = [rule for rule in self.grammar.productions()
                             if len(rule.rhs()) == 2]

        # map (Y, Z) -> X, from rhs to production
        self.prods = defaultdict(list)
        [self.prods[(str(prod.rhs()[0]), str(prod.rhs()[1]))].append(prod)
         for prod in self.double_rules]

    def parse(self, sent, lex=[]):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        # initialize
        n = len(sent)
        self._pi = pi = defaultdict(lambda: defaultdict(float))
        self._bp = bp = defaultdict(lambda: dict())

        for i in range(1, n + 1):
            for rule in self.single_rules:
                if str(rule.rhs()[0]) == sent[i - 1]:
                    rule_lhs = str(rule.lhs())
                    pi[(i, i)][rule_lhs] = rule.logprob()
                    bp[(i, i)][rule_lhs] = Tree(rule_lhs, [str(rule.rhs()[0])])

        prods = self.prods
        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                for s in range(i, j):
                    for Y, Z in product(pi[(i, s)].keys(), pi[(s + 1, j)].keys()):
                        if (Y, Z) in prods:
                            for prod in prods[(Y, Z)]:
                                score = prod.logprob() + pi[(i, s)][Y] + pi[(s + 1, j)][Z]
                                X = str(prod.lhs())
                                if X in pi[(i, j)]:
                                    if score > pi[(i, j)][X]:
                                        pi[(i, j)][X] = score
                                        bp[(i, j)][X] = Tree(X, [bp[(i, s)][Y], bp[(s + 1, j)][Z]])
                                else:
                                    pi[(i, j)][X] = score
                                    bp[(i, j)][X] = Tree(X, [bp[(i, s)][Y], bp[(s + 1, j)][Z]])

        if self._start in bp[(1, n)]:
            result = bp[(1, n)][self._start]
            return pi[(1, n)][self._start], result
        else:
            trees = [bp[(i, i)].popitem()[1] for i in range(1, n + 1)]
            return 0.0, Tree(self._start, trees)
