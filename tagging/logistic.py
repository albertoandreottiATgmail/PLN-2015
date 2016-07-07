
from collections import defaultdict
from gensim import models
from logistic_regression import LogisticRegression
from random import random
from collections import defaultdict
import theano.tensor as T
import theano
import numpy


class LogisticTagger:

    # this should be a lambda, but lambdas cannot be pickled!
    def inc(self):
        self.tag_count = self.tag_count + 1
        return self.tag_count

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        self.vec_len = vec_len = 300
        self.ending = numpy.ndarray(shape = (self.vec_len, ))
        self.ending.fill(0.9)

        self.start = numpy.ndarray(shape = (self.vec_len, ))
        self.start.fill(0.1)

        # TODO: if a trained model is found, don't train
        # map to vectors
        self.model = model = models.Word2Vec.load_word2vec_format('/home/jose/Downloads/sbw_vectors.bin', binary = True)

        self.n = b = a = n = 1
        self.tag_count = 0
        train_x , test_x , valid_x = [], [], []
        train_y , test_y , valid_y = [], [], []


        # dictionary tag -> number
        tag_number = defaultdict(self.inc)

        for sent in tagged_sents:
            psent = [('<s>', '<s>')] * b + sent + [('</s>', '</s>')] * a
            for i in range(len(sent)):
                datapoint = numpy.empty([0])
                for word, tag in psent[i: i + a + b + 1]:
                    datapoint = numpy.concatenate((datapoint, self.embedding(word)), axis = 0)

                # register the tag, assign a number if this is the first time we see it
                target = tag_number[psent[i + b][1]]

                rnd = random()
                if rnd < 0.7:
                    train_x.append(datapoint)
                    train_y.append(target)
                elif rnd < 0.8:
                    valid_x.append(datapoint)
                    valid_y.append(target)
                else:
                    test_x.append(datapoint)
                    test_y.append(target)

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized consecutive vectors.
        dataset = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

        # construct the logistic regression class
        # Each vector of embeddings has 300 elements
        classifier = LogisticRegression(dataset, x, n_in = 300 * (a + b + n), n_out = len(tag_number) + 1)
        classifier.sgd_optimization_ancora()

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def embedding(self, word):
        if word == '<s>':
            return self.start
        if word == '</s>':
            return self.ending
        if word in self.model:
            embedding = self.model[word]
            return embedding
        else:
            return [0.5] * self.vec_len

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self.most_likely
        else:
            return self.ml_tag[w]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.counts

class LazyArray(numpy.ndarray):

    def __init__(self, chunks):
        self.chunks = chunks
        self.dtype = theano.config.floatX

    def __getitem__(self, key):
        sub_array = key / 300
        return self.chunks[sub_array][key % 300]
