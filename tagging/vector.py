import pickle
from collections import defaultdict
from gensim import models
from logistic_regression import LogisticRegression
from mlp import MLP
from random import random
from collections import defaultdict
import numpy


# handles the vector representation of words
class VectorTagger:

    # this should be a lambda, but lambdas cannot be pickled!
    def inc(self):
        temp = self.tag_count
        self.tag_count += 1
        return temp

    def __init__(self, classifier, tagged_sents, window):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        self.missing = dict()
        vector_models = {'logreg': LogisticRegression, 'mlp': MLP}
        self.vec_len = vec_len = 300
        self.ending = numpy.ndarray(shape = (self.vec_len, ))
        self.ending.fill(0.9)

        self.start = numpy.ndarray(shape = (self.vec_len, ))
        self.start.fill(0.1)
        self.model = model = models.Word2Vec.load_word2vec_format('/home/jose/Downloads/sbw_vectors.bin', binary=True)
        self.window = window

        self.tag_count = 0
        train_x , test_x , valid_x = [], [], []
        train_y , test_y , valid_y = [], [], []
        # keep track of the number of sentences
        train_cnt = valid_cnt = test_cnt = 0

        # dictionary tag -> number
        tag_number = defaultdict(self.inc)

        shuffle = self.load_shuffle(len(tagged_sents))

        # map to vectors
        for i, sent in enumerate(tagged_sents):
            psent = [('<s>', '<s>')] * window.before + sent + [('</s>', '</s>')] * window.after
            tags = []
            embeddings = []
            for word, tag in psent:
                datapoint = numpy.empty([0])
                embeddings.append(numpy.concatenate((datapoint, self.embedding(word, tag)), axis=0))
                tags.append(tag_number[tag])

            rnd = shuffle[i]
            if rnd < 0.7:
                [train_x.append(embedding) for embedding in embeddings]
                [train_y.append(target) for target in tags]
                train_cnt += 1
            elif rnd < 0.8:
                [valid_x.append(embedding) for embedding in embeddings]
                [valid_y.append(target) for target in tags]
                valid_cnt += 1
            else:
                [test_x.append(embedding) for embedding in embeddings]
                [test_y.append(target) for target in tags]
                test_cnt += 1

        dataset = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

        # save the data
        with open('dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print('Number of sentences in datasets train: %d, test: %d, validation: %d' % (train_cnt, valid_cnt, test_cnt))

        # Construct the actual model class, each vector of embeddings has 300 elements
        # TODO: if a trained model is found, don't train
        # classifier = vector_models[classifier](dataset, n_in = vec_len,
        #  n_out = len(tag_number), window = window)

        classifier = MLP(dataset[0:3], n_in=300, n_hidden=120, n_out=len(tag_number), window=window)

        # clean this stuff so GC is triggered
        self.model = None
        classifier.sgd_optimization_ancora()

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def embedding(self, word, tag):
        if word == '<s>':
            return self.start
        if word == '</s>':
            return self.ending
        if word in self.model:
            embedding = self.model[word]
            return embedding
        else:
            # handle unknown
            if word not in self.missing:
                self.missing[word] = [random() for a in range(300)]
            return self.missing[word]

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

    def load_shuffle(self, nsent):
        try:
            return pickle.load(open('shuffle.pkl', 'rb'))
        except:
            shuffle = [random() for _ in range(nsent)]
            with open('shuffle.pkl', 'wb') as f:
                pickle.dump(shuffle, f)
            print('saved shuffle len: %d' % nsent)

            return shuffle




