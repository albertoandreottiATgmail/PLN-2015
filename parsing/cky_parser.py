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

        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                for X_rule in self.grammar.productions():
                    if len(X_rule.rhs()) < 2:
                        continue
                    X = str(X_rule.lhs())

                    candidates = []
                    Z = str(X_rule.rhs()[0])
                    Y = str(X_rule.rhs()[1])

                    for s in range(i, j):
                        if Z in pi[(i, s)] and Y in pi[(s + 1, j)]:
                            candidates.append((X_rule.logprob() + pi[(i, s)][Z] + pi[(s + 1, j)][Y], s))

                    if len(candidates) == 0:
                        continue

                    top = max(candidates, key=lambda x: x[0])

                    if X in pi[(i, j)]:
                        if top[0] > pi[(i, j)][X]:
                            pi[(i, j)][X] = top[0]
                            bp[(i, j)][X] = (X_rule, top[1])
                    else:
                        pi[(i, j)][X] = top[0]
                        bp[(i, j)][X] = (X_rule, top[1])

        def populate_tree(i, j, lhs):
            if i != j:
                entry = bp[(i, j)][lhs]
                return Tree(lhs, [populate_tree(i, entry[1], str(entry[0].rhs()[0])),
                            populate_tree(entry[1] + 1, j, str(entry[0].rhs()[1]))])
            else:
                # use i,j to fetch the word from the sentence in the upcfg case
                if self._is_lexicalized:
                    return Tree(lhs, [str(bp[(i, j)][lhs][0].rhs()[0])])
                else:
                    return Tree(lhs, [lex[i - 1]])

        if self._start in pi[(1, n)]:
            tree = populate_tree(1, n, self._start)
            tree.un_chomsky_normal_form()
            return pi[(1, n)][self._start], tree
        else:
            children = [Tree(tag, [word]) for tag, word, in zip(sent, lex)]
            return 0.0, Tree(self._start, children)
