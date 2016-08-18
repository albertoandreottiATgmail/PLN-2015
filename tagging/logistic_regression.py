"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d
from data_access import DataAccess
from util import Window


class LogisticRegression(DataAccess):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, dataset, n_in, n_out, window=Window(0,0), x=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type wsize: int
        :param n_out: context, size of the window, whose center is the vector
                      we're trying to classify

        """

        print ('window size: %d:%d' % (window.before, window.after))
        self.dataset = dataset
        self.wsize = wsize = window.before + 1 + window.after
        self.window = window

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        if x is None:
            x = T.matrix('x')  # data, each vector of matrix is an embedding for a word.
        # keep track of model input
        self.input = x

        # initialize with 0 the weights W as a matrix of shape (n_in, wsize, n_out)
        # Added an extra dimension, wsize. A vector in the old W is divided wsize times,
        # and each chunk is located along that extra dimension.
        # Nothing changes from a mathematical point of view, just a more convenient layout.
        self.W = theano.shared(
            value=numpy.zeros(
                (n_out, wsize, n_in),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a tensor where sub-matrix k represent the separation hyperplane for class-k.
        # x is a matrix, where each set of wsize consecutive rows represents an input training sample.
        # b is a vector where element-k represent the free parameter of hyperplane-k.
        projections = conv2d(x, self.W)
        projections = theano.tensor.addbroadcast(projections, 2).squeeze().transpose()
        self.p_y_given_x = T.nnet.softmax(projections + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def sgd_optimization_ancora(self, learning_rate=0.22, n_epochs=250,
                                batch_size=300):
        """
        Stochastic gradient descent optimization of the log-linear model

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type batch_size: int
        :param batch_size: size of dataset over which average is taken

        """

        train_set_x, train_set_y = self.shared_dataset(self.dataset[0])
        valid_set_x, valid_set_y = self.shared_dataset(self.dataset[1])
        test_set_x, test_set_y = self.shared_dataset(self.dataset[2])
        self.dataset = None

        # compute number of mini-batches for training, validation and testing
        n_train_batches = train_set_y.shape.eval()[0] // batch_size
        n_valid_batches = valid_set_y.shape.eval()[0] // batch_size
        n_test_batches = test_set_y.shape.eval()[0] // batch_size

        print('... building the model')

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = self.input
        y = T.ivector('y')   # labels, presented as 1D vector of [int] labels

        # the cost we minimize during training is the negative log likelihood of the model
        cost = self.negative_log_likelihood(y)
        window = self.window

        # compiling a Theano function that computes the
        # mistakes that are made by the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=self.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size + window.after + window.before],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size + window.after + window.before],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.W, self.W - learning_rate * g_W),
                   (self.b, self.b - learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size + window.after + window.before],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995   # a relative improvement of this much is
                                        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()
        valid_losses = []

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number(in number of batches)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    valid_losses.append(this_validation_loss)
                    print('validation loss: ' + str(this_validation_loss))

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss

                        # test it on the test set
                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print((' epoch %i, minibatch %i/%i, test error of'
                               ' best model %f %%') %
                              (epoch,
                               minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

                        # save the best model
                        # with open('best_model.pkl', 'wb') as f:
                        #    pickle.dump(self, f)

                """ Validation Set: this data set is used to minimize overfitting.
                You're not adjusting the weights of the network with this data set,
                you're just verifying that any increase in accuracy over the training
                data set actually yields an increase in accuracy over a data set that
                has not been shown to the network before, or at least the network hasn't
                trained on it (i.e. validation data set). If the accuracy over the
                training data set increases, but the accuracy over then validation data
                set stays the same or decreases, then you're overfitting your neural network
                and you should stop training."""

                if patience <= iter:
                    done_looping = True
                    break
        # save validation losses
        with open('losses.pkl', 'wb') as f:
            pickle.dump(valid_losses, f)

        end_time = timeit.default_timer()
        print(('Optimization complete with best validation score of %f %%,'
              'with test performance %f %%')
              % (best_validation_loss * 100., test_score * 100.))

        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))

        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

    def predict(self):
        """
        An example of how to load a trained model and use it
        to predict labels.
        """

        # load the saved model
        classifier = pickle.load(open('best_model.pkl', 'rb'))

        # compile a predictor function
        predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.y_pred)

        # We can test it on some examples from test test
        dataset='mnist.pkl.gz'
        datasets = self.datasets
        test_set_x, test_set_y = datasets[2]
        test_set_x = test_set_x.get_value()

        predicted_values = predict_model(test_set_x[:10])
        print("Predicted values for the first 10 examples in test set:")
        print(predicted_values)


if __name__ == '__main__':
    # sgd_optimization_ancora()
    # predict()
    pass
