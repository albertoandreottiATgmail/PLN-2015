# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from nltk.tree import Tree
from nltk.grammar import Nonterminal as N, ProbabilisticProduction

from parsing.upcfg import UPCFG


class TestUPCFG(TestCase):

    def test_init_doesnt_change_trees(self):
        t = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        t2 = t.copy(deep=True)

        model = UPCFG([t])

        self.assertEqual(t, t2)

    def test_productions(self):
        t = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)

        model = UPCFG([t])

        prods = model.productions()

        prods2 = [
            ProbabilisticProduction(N('S'), [N('NP'), N('VP')], prob=1.0),
            ProbabilisticProduction(N('NP'), [N('Det'), N('Noun')], prob=0.5),
            ProbabilisticProduction(N('Det'), ['Det'], prob=1.0),
            ProbabilisticProduction(N('Noun'), ['Noun'], prob=1.0),
            ProbabilisticProduction(N('VP'), [N('Verb'), N('NP')], prob=1.0),
            ProbabilisticProduction(N('Verb'), ['Verb'], prob=1.0),
            ProbabilisticProduction(N('NP'), [N('Noun'), N('Adj')], prob=0.5),
            ProbabilisticProduction(N('Adj'), ['Adj'], prob=1.0),
        ]

        self.assertEqual(set(prods), set(prods2))

    def test_parse(self):
        t = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        model = UPCFG([t], start='S')

        sent = 'el gato come pescado crudo'.split()
        tags = 'Det Noun Verb Noun Adj'.split()
        tagged_sent = list(zip(sent, tags))
        tree = model.parse(tagged_sent)
        tree.pretty_print()
        t.pretty_print()
        self.assertEqual(tree, t)

    def test_parse_no_parse_returns_flat(self):
        t = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        model = UPCFG([t], start='S')


        sent = 'gato el come pescado crudo'.split()
        tags = 'Noun Det Verb Noun Adj'.split()
        tagged_sent = list(zip(sent, tags))
        tree = model.parse(tagged_sent)

        tree2 = Tree.fromstring("(S (Noun gato) (Det el) (Verb come) (Noun pescado) (Adj crudo))")
        tree.pretty_print()
        tree2.pretty_print()
        self.assertEqual(tree, tree2)
