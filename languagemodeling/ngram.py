# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import pow, log
from functools import reduce
from random import random
from languagemodeling.scripts.leipzigreader import LeipzigCorpusReader
import pyprind

log2 = lambda x: log(x, 2.0)
pow2 = lambda x: pow(2.0, x)

def compute_ngram(test_set, model):
    M = 0
    log_probability = 0.0
    for sentence in test_set.sents():
        sentence.append('</s>')
        sentence.insert(0, '<s>')
        M += len(sentence)
        prev_tokens = sentence[ : model.n - 1]
        sent_prob = log2(model.cond_prob(sentence[model.n - 1], prev_tokens))
        for word in sentence[model.n - 1 : ]:
            sent_prob += log2(model.cond_prob(word, prev_tokens))
            prev_tokens = prev_tokens[1:] + [word]
        log_probability += sent_prob

    return (log_probability, M)

def compute_1gram(test_set, model):

    M = 0
    log_probability = 0.0
    for sentence in test_set.sents():
        sentence.append('</s>')
        M += len(sentence)
        sent_prob = model.cond_prob(sentence[model.n - 1], None)
        for word in sentence[model.n - 1 : ]:
            sent_prob *= model.cond_prob(word, None)
        log_probability += log2(sent_prob)

    return (log_probability, M)




class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        lsents = []
        for sent in sents:
            s = list(sent)
            if n > 1:
                s.insert(0, '<s>')
            s.append('</s>')
            lsents.append(s)

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
        
        if prev_tokens == None or self.n == 1: 
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
        lsents = []
        for sent in sents:
            s = list(sent)
            if n > 1:
                s.insert(0, '<s>')
            s.append('</s>')
            lsents.append(s)
        
        vocabulary = defaultdict(int)
        for sent in lsents:
            for word in sent:
                vocabulary[word] += 1
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
        self._V = len(vocabulary) 
        
        if n > 1: 
            self._V -= 1 #remove count for <s>       
        
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        
        #local alias for counts
        counts = self.counts
        
        if prev_tokens == None:
            return float(counts[tuple([token])] + 1) / (counts[()] + self._V)

        local_prev_tokens = list(prev_tokens)
        local_prev_tokens.append(token)
        num = counts[tuple(local_prev_tokens)]

        denom = counts[tuple(prev_tokens)]
        try:
            return float(num + 1) / (denom + self._V)
        except ZeroDivisionError:
            return float('inf')
    def V(self):
        return self._V

class InterpolatedNGram(NGram):
    
    def __init__(self, n, sents, gamma=None, addone=True):    
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        
        if addone:
            self.models = [AddOneNGram(i + 1, sents) for i in range(n)]
        else:
            self.models = [NGram(i + 1, sents) for i in range(n)]

        #if gamma is given, do not train
        if gamma or n == 1:
            self.gamma = gamma
            return
        gamma_range = self.drange(85, 125, 0.5)
        held_out = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_train_held_out')

        top_gamma = {'gamma' : gamma, 'perplexity' : 10000000.0}
        model = InterpolatedNGram(n, sents, 0.01)
        for g in gamma_range:
            
            model.gamma = g
            (logp, M) = compute_ngram(held_out, model)
            cross_entropy = -logp / float(M)
            perplexity = pow2(cross_entropy)
            if top_gamma['perplexity'] > perplexity:
                top_gamma['perplexity'] = perplexity
                top_gamma['gamma'] = g
        print(top_gamma)    
        self.gamma = top_gamma['gamma']    
         

    def drange(self, start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step

            
    def computeLambdas(self, gamma, counts):

        assert(len(counts) == self.n - 1)
        #lambdas[1] is lambda1.
        lambdas = {k + 1: 0 for k in range(len(counts))}
        for k in lambdas:
            lambdas[k] = (1 - sum([lambdas[i + 1] for i in range(k)])) * counts[k - 1] / (counts[k - 1] + gamma)
        lambdas[self.n] = (1 - sum([lambdas[i + 1] for i in range(self.n - 1)]))
        return lambdas    

    def count(self, tokens):
        """Count for a k-gram for k <= n.
 
        tokens -- the k-gram tuple.
        """
        if len(tokens) == 0:
            model = 0
        else:
            model = len(tokens) - 1

        return self.models[model].count(tuple(tokens))    
 
    def cond_prob(self, token, prev_tokens=[]):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        gamma = self.gamma

        if n == 1:
            return self.models[0].cond_prob(token)

        ngram = tuple(prev_tokens + [token])
        ngrams = [tuple(ngram[i:-1]) for i in reversed(range(n - 1))]
        counts = [self.models[i].count(ngrams[i]) for i in range(n - 1)]
        assert(len(self.models) == n)
        qmls = [m.cond_prob(ngram[-1], ng) for (m, ng) in zip(self.models, [None] + ngrams)]
        
        #for x in zip(self.models, [None] + ngrams):
        #    print('zip: ', x[0].n, x[1])
        #print('qmls: ', qmls)

        lambdas = self.computeLambdas(gamma, counts)
        #print(lambdas)
        assert(len(counts) + 1 == len(lambdas))
        assert(len(lambdas) == n)

        lambda_list = []
        for k in sorted(lambdas.keys() ):
            lambda_list.append(lambdas[k])

        return self.dot_product(reversed(lambda_list), qmls) 
    
    def dot_product(self, v, w):
        #print('v: ', v, '/n', 'w: ', w)
        return sum(map(lambda x: x[0] * x[1], zip(v, w)))




class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """

        self.n = n
        self.models = {}

        if n == 1:
            self.models[1] = AddOneNGram(1, sents) if addone else NGram(1, sents)
            print(self.models[1].counts)
        else:

            lsents = []
            for sent in sents:
                s = list(sent)
                if n > 1:
                    s.insert(0, '<s>')
                s.append('</s>')
                lsents.append(s)


            self.sents = lsents
            held_out = LeipzigCorpusReader('eng-za_web_2013_100K-sentences.txt_train_held_out')
            self.counts = defaultdict(int)
            for sent in lsents:
                for i in range(len(sent) - n + 1):
                    ngram = tuple(sent[i: i + n])
                    self.counts[ngram] += 1
                    self.counts[ngram[:-1]] += 1
            self.counts[tuple(['</s>'])] = len(lsents)            
            print(self.counts)
            if n == 2:
                self.models[n - 1] = AddOneNGram(n - 1, sents) if addone else NGram(n - 1, sents)
            else:
                self.models[n - 1] = BackOffNGram(n - 1, sents)

            top_beta = {'beta' : 0, 'perplexity' : 1000000.0}

            if beta != None:
                self.beta = beta
                return

            print('training for N: %d' % self.n)
            for b in pyprind.prog_bar(list(self.drange(0.1, 0.7, 0.5))):
                self.beta = b
                (logp, M) = compute_ngram(held_out, self)
                cross_entropy = -logp / float(M)
                perplexity = pow2(cross_entropy)
                if top_beta['perplexity'] > perplexity:
                    top_beta['perplexity'] = perplexity
                    top_beta['beta'] = b
        
            self.beta = float(top_beta['beta'])


    def count(self, ngram):
        if self.n == 1:
            return self.models[1].count(ngram)
        else:
            return self.counts[tuple(ngram)]
 
    def cond_prob(self, token, prev_tokens=None):
        if self.n == 1 or prev_tokens == None: 
            return self.models[1].cond_prob(token)

        print ('A():' , self.A(tuple(prev_tokens)), 'token', token, 'prev_tokens', prev_tokens)
        if token in self.A(tuple(prev_tokens)):
            print('token:',  token, 'prev_tokens', prev_tokens, 'prob', (self.counts[tuple(prev_tokens + [token])] - self.beta) / self.counts[tuple(prev_tokens)])
            #print (self.counts)
            #print ('pt:', prev_tokens)
            return (self.counts[tuple(prev_tokens + [token])] - self.beta) / self.counts[tuple(prev_tokens)]
        else:
            print('alpha', self.alpha(tuple(prev_tokens)), 'n-1: ', self.models[self.n - 1].cond_prob(token, prev_tokens[1:]), 'denom:', self.denom(tuple(prev_tokens)))
            return self.alpha(tuple(prev_tokens)) * self.models[self.n - 1].cond_prob(token, prev_tokens[1:]) / self.denom(tuple(prev_tokens))

    def drange(self, start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step
        
    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        #print(self.n, tokens)
        answer = set([])
        assert(len(tokens) < self.n)
        #print (self.sents)
        for sent in self.sents:
            for i in range(len(sent) - self.n + 1):
                ngram = sent[i: i + self.n]
                if tuple(ngram[:-1]) == tokens:
                    answer.add(ngram[-1])

        return answer
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        acc = 0.0
        for w in self.A(tokens):
            acc += (self.counts[tuple(tokens) + tuple([w])] - self.beta) / self.counts[tuple(tokens)]
        return 1 - acc

 
    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        acc = 0.0
        for x in self.A(tokens):
            acc += self.models[self.n - 1].cond_prob(x, list(tokens[1:]))
        print('acc', acc)
        return 1 - acc    
