# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                NP -> Det Noun          [0.6]
                NP -> Noun Adj          [0.4]
                VP -> Verb NP           [1.0]
                Det -> 'el'             [1.0]
                Noun -> 'gato'          [0.9]
                Noun -> 'pescado'       [0.1]
                Verb -> 'come'          [1.0]
                Adj -> 'crudo'          [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     log2(1.0) +  # rule S -> NP VP
                     log2(0.6 * 1.0 * 0.9) +  # left part
                     log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},  # right part
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},
            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},
            (3, 5): {'VP': Tree.fromstring("(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},
            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }

        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def test_ambiguous_parse(self):
        
        sentence = 'I drove down the road in the car'

        # rely on : (VP -> VP PP[.5]) > (NP -> NP PP[0.45])
        # replace VP -> VB PP             [.5]
        #         VP -> VP PP             [.5]
        # with    VP -> VB PP             [.6]
        #         VP -> VP PP             [.4]
        # to obtain the wrong parse tree.

        correct_grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                VP -> VB PP             [.5]
                VP -> VP PP             [.5]
                PP -> IN NP             [1.0]
                NP -> NP PP             [0.45] 
                NP -> Det NN            [0.45]
                NP -> I                 [0.1]
                VB -> drove             [1.0]
                IN -> down              [0.5]
                IN -> in                [0.5]
                Det -> the              [1.0]
                NN -> road            [.5]
                NN -> car             [.5]
            """)

        parser = CKYParser(correct_grammar)
        tree =  Tree.fromstring(
            """
                (S
                    (NP I)
                    (VP
                         (VP (VB drove) (PP (IN down) (NP (Det the) (NN road))))
                         (PP (IN in) (NP (Det the) (NN car)))
                    )
                )
            """)
        tree.pretty_print()
        parsed = parser.parse(sentence.split())[1]
        parsed.pretty_print()
        self.assertEqual(parsed, tree)


    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
