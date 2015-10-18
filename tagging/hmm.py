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
        sent = y + ['</s>']
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
            poutput *= self.out[y[i]][x[i]]

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


class MLHMM(HMM):
 
    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.tag_set = tag_set = set()
        self.counts = counts = defaultdict(int)
        self.out = out = defaultdict(lambda: defaultdict(int)) 
        self.trans  = trans = defaultdict(lambda: defaultdict(float))
        self.vocab = set() 

        for sent in tagged_sents:
            padded_sent = sent + [('</s>', '</s>')]
            if n > 1:
                padded_sent = [('<s>', '<s>')] + padded_sent

            for i in range(len(padded_sent) - n + 1):
                [words, tags] = zip(*padded_sent[i: i + n])
                tag_set.add(tags[0])

                #TODO: refactor this!
                counts[tags] += 1
                counts[tags[:-1]] += 1

                out[tags[0]][words[0]] += 1
                trans[tags[:-1]][tags[-1]] += 1
                self.vocab.add(words[0])

                if i == (len(padded_sent) - n) and n > 1:
                    counts[tuple(words[1:])] += 1
                    [self.vocab.add(x) for x in words[1:]]
                    [self.tag_set.add(x) for x in tags[1:]]

        for tag in out:
            for word in out[tag]:
                self.out[tag][word] /= counts[(tag, )]
        
        for prev in trans:
            for tag in trans[prev]:
                self.trans[prev][tag] /= counts[prev] 

  
    def tcount(self, tokens):
        """Count for an n-gram or (n-1)-gram of tags.
 
        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        return self.counts[tokens]
 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        return w not in self.vocab
 

    """
       Todos los métodos de HMM.
    """

 
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
        hmm = self.hmm
        S = lambda k: set(['<s>']) if k < 1 else hmm.tagset().union(set(['<s>'])) 
        m = len(sent)

        #response
        y = (m + 1) * [0.0]

        #initialize pi[k=0]
        pi[0][tuple((hmm.n - 1) * ['<s>'])] = 1.0

        for k in range(1, m + 1):
            for end_tokens in product(*[S(k) for k in range(k - hmm.n + 2, k + 1)]):
                candidates = [(w, pi[k - 1][(w, ) + end_tokens[:-1]] * hmm.trans_prob(end_tokens[-1], (w, ) + end_tokens[:-1]) * hmm.out_prob(sent[k - 1], end_tokens[-1])) for w in S(k - hmm.n + 1)]
                pi[k][end_tokens] = max(candidates, key = lambda x: x[1])[1] 
                bp[k][end_tokens] = max(candidates, key = lambda x: x[1])[0] # [0] is argmax

        #build response, tri-gram case 
        #(y[m - 1], y[m]) = max([(pi[m][(u,v)] * hmm.trans_prob('</s>', (u, v)) , (u, v)) for u in hmm.tagset() for v in hmm.tagset()])[1]
        
        tuples = product(*[S(k) for k in range(k - hmm.n + 2, k + 1)])
        y[m - hmm.n + 2 : m + 1] = max([(pi[m][prev] * hmm.trans_prob('</s>', prev) , prev) for prev in tuples])[1]

        for k in reversed(list(range(m - hmm.n + 2))):
            y[k] = bp[k + hmm.n - 1][tuple(y[k + 1 : k + hmm.n])]

        return y[1:]    
