# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import pow, log
from functools import reduce
from random import random

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
        
        #local alias for counts
        counts = self.counts
        
        if prev_tokens == None: 
            return float(counts[tuple([token])]) / counts[()]

        local_prev_tokens = list(prev_tokens)
        local_prev_tokens.append(token)
        num = counts[tuple(local_prev_tokens)]
       
        #If we've never seen this ngram, just return 0.0
        if num == 0.0:
            return num

        denom = counts[tuple(prev_tokens)]

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

        return sum(probabilities)

class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        
        self._model = model

        #short_ngrams = filter(lambda x: len(x) == model.n - 1, model.counts)
        self._sampling_model = defaultdict(list)
        #for short_ngram in short_ngrams:
        #    long_ngrams = filter(lambda x: len(x) == model.n and short_ngram == x[0 : -1], model.counts)
        #    current_step_beginning = 0.0
        #    self._sampling_model[short_ngram]
        #    for long_ngram in long_ngrams:
        #        current_step = model.cond_prob(long_ngram[-1], long_ngram[:-1])
        #        self._sampling_model[short_ngram].append((current_step_beginning, current_step_beginning + current_step, long_ngram[-1]))
        #        current_step_beginning += current_step
            #this must be a distribution!
        #    assert(abs(current_step_beginning - 1.0) < 0.0001)

    def fill_cache(self, short_ngram):
        model = self._model

        if tuple(short_ngram) in self._sampling_model:
            return

        long_ngrams = filter(lambda x: len(x) == model.n and short_ngram == x[0 : -1], model.counts)
        current_step_beginning = 0.0
        self._sampling_model[tuple(short_ngram)]

        for long_ngram in long_ngrams:
            current_step = model.cond_prob(long_ngram[-1], long_ngram[:-1])
            self._sampling_model[short_ngram].append((current_step_beginning, current_step_beginning + current_step, long_ngram[-1]))
            current_step_beginning += current_step
        #this must be a distribution!
        assert(abs(current_step_beginning - 1.0) < 0.0001)
            
    def generate_sent(self):
        """Randomly generate a sentence."""

        #generate random number in [0,1], look for N-gram mapped to interval
        sample = random()
        #basic case for unigrams
        prev_tokens = tuple(['<s>'])
        result = ['<s>']
        model = self._model

        #if this is a model other than unigram, let's sample to obtain 'prev_tokens'
        if model.n > 1:
            start_ngrams = [(ngram, model.counts[ngram]) for ngram in model.counts if len(ngram) == model.n and ngram[0] == '<s>']
            total = sum(map(lambda x: x[1], start_ngrams))
            current_pos = 0.0
            intervals = []
            for ngram in start_ngrams:
                step = float(ngram[1]) / total
                intervals.append((ngram[0], current_pos, current_pos + step))
                current_pos += step
            #TODO: replace this inefficient linear search
            for ngram in intervals:
                if ngram[1] < sample < ngram[2]:
                    prev_tokens = ngram[0]
                    result = list(prev_tokens)

        sampled = None
        while sampled != '</s>':
            sampled = self.generate_token(prev_tokens[1:])
            prev_tokens = prev_tokens[1:] + (sampled,)
            result.append(sampled)
        return result[1:-1]    


    def generate_token(self, p_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        sample = random()
        result  = None
        #for ngram in self._sampling_model[p_tokens]:
        #    if ngram[0] < sample < ngram[1]:
        #        result = ngram[2]
        #return result
        if p_tokens == None:
            self.fill_cache(tuple())
        else:    
            self.fill_cache(tuple(p_tokens))        

        def binary_search(chunk):
            element = chunk[int(len(chunk) / 2)]
            if element[0] < sample < element[1]:
                return element
            elif sample < element[0]:
                return binary_search(chunk[0 : int(len(chunk) / 2)])    
            else:     
                return binary_search(chunk[int(len(chunk) / 2) + 1:])    

        return binary_search(self._sampling_model[p_tokens])[2]


class AddOneNGram(NGram):

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
        
        vocabulary = defaultdict(int)
        for sent in lsents:
            for word in sent:
                vocabulary[word] += 1
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
        self._V = len(vocabulary) + 2 # for <s> and </s>       
        
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        
        #local alias for counts
        counts = self.counts
        
        if prev_tokens == None: 
            return float(counts[tuple([token])]) / counts[()]

        local_prev_tokens = list(prev_tokens)
        local_prev_tokens.append(token)
        num = counts[tuple(local_prev_tokens)]
       
        #If we've never seen this ngram, just return 0.0
        if num == 0.0:
            return num

        denom = counts[tuple(prev_tokens)]
        try:
            return float(num + 1) / (denom + self._V)
        except ZeroDivisionError:
            return float('inf')
