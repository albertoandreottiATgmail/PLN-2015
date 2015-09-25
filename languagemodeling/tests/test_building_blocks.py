# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from languagemodeling.ngram import BackOffNGram


class TestInterpolatedNGram(TestCase):

    def setUp(self):
        self.sents = [
            'el gato come pescado .'.split(),
            'la gata come salmón .'.split(),
        ]

    def test_A_bigram(self):
        model = BackOffNGram(2, self.sents, beta=0.5, addone=False)

        self.assertEqual(model.A(tuple(['gato'])), set(['come']))
        self.assertEqual(model.A(tuple(['gata'])), set(['come']))
        self.assertEqual(model.A(tuple(['come'])), set(['salmón', 'pescado']))
        self.assertEqual(model.A(tuple(['come'])), set(['salmón', 'pescado']))


    def test_A_trigram(self):
        model = BackOffNGram(3, self.sents, beta=0.5, addone=False)


        self.assertEqual(model.A(['gato', 'come']), set(['pescado']))
        self.assertEqual(model.A(['la', 'gata']), set(['come']))
