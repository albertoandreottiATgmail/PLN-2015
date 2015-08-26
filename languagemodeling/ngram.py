# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import pow, log

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]


    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tuple(tokens)]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if self.n is not 1 and prev_tokens is None:
            return 0.0 # TODO: throw!

        counts = self.counts
        if prev_tokens == None:
            total = lambda z: True
            num = counts[tuple(token)]
        else:
            total = lambda z: z[:-1] == tuple(prev_tokens)
            local_prev_tokens = list(prev_tokens)
            local_prev_tokens.append(token[0])
            num = counts[tuple(local_prev_tokens)]
        
        
        keys = filter(total, counts)
        print keys
        denom = reduce(lambda x, y: x + y, map(lambda key: counts[key], keys))
        return float(num) / denom
 
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        pow(2, self.sent_log_prob(sent))

 
    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        probabilities = [self.cond_prob(sent[i : i + self.n - 1], sent[i + self.n]) for i in range(len(sent) - 1)]
        #append beginning and end
        #probabilities.append()
        #probabilities.append()
        probabilities = [log(prob, 2) for prob in probabilities]
        reduce(lambda x, y:  x + y, probabilities)




