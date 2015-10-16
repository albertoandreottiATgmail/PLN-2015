from math import log
from itertools import product
from collections import defaultdict

log2 = lambda x: log(x, 2.0)

class HMM:
 
    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- tag transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tag_set = tagset
        self.trans = defaultdict(lambda: defaultdict(int)) 
        self.trans.update(trans)
        
        self.out = defaultdict(lambda: defaultdict(int)) 
        self.out.update(out)
 
    def tagset(self):
        """Returns the set of tags.
        """
        return self.tag_set
 
    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
 
        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        return self.trans[prev_tags][tag] if tag in self.trans[prev_tags] else 0.0

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
 
        word -- the word.
        tag -- the tag.
        """
        return self.out[tag][word] if word in self.out[tag] else 0.0
 
    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.
 
        y -- tagging.
        """
        sent = ['<s>'] + y + ['</s>']
        prob = 1.0
        for i in range(len(sent) - self.n + 1):
            prob *= self.trans[tuple(sent[i:i + self.n - 1])][sent[i + self.n - 1]]

        return prob    

 
    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.
 
        x -- sentence.
        y -- tagging.
        """
        poutput = 1.0
        for i in range(len(x)):
            print(x[i])
            poutput *= self.out[y[i]][x[i]]

        print(poutput)
        return self.tag_prob(y) * poutput

 
    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.
 
        y -- tagging.
        """
        return log2(self.tag_prob(y))

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.
 
        x -- sentence.
        y -- tagging.
        """
        return log2(self.prob(x,y))
 
    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
 
        sent -- the sentence.
        """
        return ViterbiTagger(self).tag(sent)
 
 
class ViterbiTagger:
 
    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm
 
    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
 
        sent -- the sentence.
        """
        pi = [defaultdict(float) for i in range(len(sent) + 2)]
        bp = [defaultdict(str) for i in range(len(sent) + 2)]
        S = lambda k: set(['<s>']) if k < 1 else hmm.tagset().union(set(['<s>'])) 

        hmm = self.hmm
        m = len(sent)

        #response
        y = (m + 1) * [0.0]

        #initialize pi[k=0]
        pi[0][('<s>', '<s>')] = 1.0

        for k in range(1, m + 1):
            for u, v in product(S(k - 1), S(k)):
                candidates = [(w, pi[k - 1][(w,u)] * hmm.trans_prob(v, (w, u)) * hmm.out_prob(sent[k - 1], v)) for w in S(k - 2)]
                assert(pi[0][('<s>', '<s>')] == 1.0)
                pi[k][(u,v)] = max(candidates, key = lambda x: x[1])[1] 
                bp[k][(u,v)] = max(candidates, key = lambda x: x[1])[0] # [0] is argmax

        #build response 
        (y[m - 1], y[m]) = max([(pi[m][(u,v)] * hmm.trans_prob('</s>', (u, v)) , (u, v)) for u in hmm.tagset() for v in hmm.tagset()])[1]
        
        for k in range(m - 1):
            y[k] = bp[k + 2][(y[k + 1], y[k + 2])]

        return y[1:]    


