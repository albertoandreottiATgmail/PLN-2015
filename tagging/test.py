import numpy
import theano
from theano.tensor.signal.conv import conv2d
# input vectors
a = numpy.ones((20, 5),
    dtype=theano.config.floatX)

ta = theano.shared(value=a,
    name='W',
    borrow=True)

# 5 dimension of the embedding vectors
# 3 window size
# 2 projections over the hyperplanes(classification classes)
b = numpy.zeros((4, 3, 5),
    dtype=theano.config.floatX)

b[0,0,0] = b[0,1,0] = b[0,2,0] = 1
b[1,0,0] = b[1,1,0] = b[1,2,0] = 2

tb = theano.shared(value=b,
    name='W',
    borrow=True)

tc = conv2d(ta, tb)
#tc.shape.eval()	
#tc.reshape([2, 18]).transpose().eval()
tc = theano.tensor.addbroadcast(tc, 2).squeeze().transpose().eval()
