# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import pow, log
from functools import reduce

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        lsents = list(sents)
        for sent in lsents:
            if n > 1:
                sent.insert(0, '<s>')
            sent.append('</s>')

        for sent in lsents:
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
        
        #local copy for counts
        counts = defaultdict(int, self.counts)
        
        counts.pop((), None)
        counts.pop('<s>', None)

        if prev_tokens == None: 
            total = lambda z: True 
            num = counts[tuple([token])]
        else:
            total = lambda z: z[:-1] == tuple(prev_tokens)
            local_prev_tokens = list(prev_tokens)
            local_prev_tokens.append(token)
            num = counts[tuple(local_prev_tokens)]
       
        #If we've never seen this ngram, just return 0.0
        if num == 0.0:
            return num

        keys = filter(total, counts)
        denom = reduce(lambda x, y: x + y, map(lambda key: counts[key], keys))
        try:
            return float(num) / denom
        except ZeroDivisionError:
            return float('inf')
         
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        return pow(2, self.sent_log_prob(sent))

 
    def sent_log_prob(self, sentence):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        sent = list(sentence)
        sent.append('</s>')
        if self.n > 1:
            sent.insert(0, '<s>')
            probabilities = [self.cond_prob(sent[i], sent[i - self.n + 1: i]) for i in range(1, len(sent))]
        else: 
            probabilities = [self.cond_prob(word) for word in sent]

        log2 = lambda x: float('-inf') if x == 0.0 else log(x, 2)
        probabilities = [log2(prob) for prob in probabilities]
        return reduce(lambda x, y:  x + y, probabilities)




