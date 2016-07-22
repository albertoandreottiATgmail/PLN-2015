import numpy
import theano
from theano.tensor.signal.conv import conv2d

a = numpy.ones((20, 5),
    dtype=theano.config.floatX)

ta = theano.shared(value=a,
    name='W',
    borrow=True)

# 5 dimension of the embedding vectors
# 3 window size
# 2 projections over the hyperplanes(classification classes)
b = numpy.zeros((2, 3, 5),
    dtype=theano.config.floatX)

b[0,0,0] = b[0,1,0] = b[0,2,0] = 1
b[1,2,0] = 1

tb = theano.shared(value=b,
    name='W',
    borrow=True)

conv2d(ta, tb).eval()
