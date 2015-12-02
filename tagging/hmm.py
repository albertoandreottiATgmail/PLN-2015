from math import log
from collections import defaultdict


def log2(x):
    return log(x, 2.0)


def dict_int():
    return defaultdict(int)


def dict_float():
    return defaultdict(float)


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
        self.trans = defaultdict(dict_int)
        self.trans.update(trans)

        self.out = defaultdict(dict_int)
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
        if tag in self.trans[prev_tags]:
            return self.trans[prev_tags][tag]
        else:
            return 0.0

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
        return log2(self.prob(x, y))

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
        self.out = out = defaultdict(dict_float)
        self.trans = trans = defaultdict(dict_float)
        self.vocab = set()

        for sent in tagged_sents:
            padded_sent = sent + [('</s>', '</s>')]

            if n > 1:
                padded_sent = (n - 1) * [('<s>', '<s>')] + padded_sent

            for i in range(len(padded_sent) - n + 1):
                [words, tags] = zip(*padded_sent[i: i + n])
                tag_set.add(tags[0])

                counts[tags] += 1
                counts[tags[:-1]] += 1
                if n > 2:
                    counts[(tags[0],)] += 1

                out[tags[0]][words[0]] += 1
                trans[tags[:-1]][tags[-1]] += 1
                self.vocab.add(words[0])

                if i == (len(padded_sent) - n) and n > 1:
                    counts[tuple(words[1:])] += 1
                    [self.vocab.add(x) for x in words[1:]]
                    [self.tag_set.add(x) for x in tags[1:]]
                    for tag, word in zip(tags[1:], words[1:]):
                        out[tag][word] += 1

        for tag in out:
            for word in out[tag]:
                try:
                    self.out[tag][word] /= float(counts[(tag, )])
                except ZeroDivisionError:
                    pass
                    print(tag, counts[(tag, )])

        for prev in trans:
            for tag in trans[prev]:
                denom = counts[prev]
                if addone:
                    self.trans[prev][tag] += 1.0
                    denom += len(self.trans)
                self.trans[prev][tag] /= denom

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

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        return self.trans[prev_tags][tag] if tag in self.trans[prev_tags] else 1.0 / len(self.trans)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        if word in self.vocab:
            return self.out[tag][word]
        else:
            return 1.0 / len(self.vocab)
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
        cand = [defaultdict(list) for i in range(len(sent) + 2)]
        bp = [defaultdict(str) for i in range(len(sent) + 2)]
        hmm = self.hmm
        tagset_ = hmm.tagset().union(set(['<s>']))
        S = lambda k: set(['<s>']) if k < 1 else tagset_
        m = len(sent)

        # response
        y = (m + 1) * [0.0]

        # initialize pi[k=0]
        if hmm.n > 1:
            pi[0][tuple((hmm.n - 1) * ['<s>'])] = log2(1.0)
        else:
            pi[0][()] = log2(1.0)

        for k in range(1, m + 1):
            for key in pi[k - 1]:
                for v in S(k):
                    out_p = hmm.out_prob(sent[k - 1], v)
                    if out_p > 0.0 and hmm.trans_prob(v, key) > 0.0:
                        end_tokens = key[1:] + (v, )
                        w = key[0] if hmm.n > 1 else v
                        value = pi[k-1][key] + log2(hmm.trans_prob(v, key)) + log2(out_p)
                        cand[k][end_tokens].append((w, value))

            for etokens in cand[k]:
                best = max(cand[k][etokens], key=lambda x: x[1])
                pi[k][etokens] = best[1]
                bp[k][etokens] = best[0]  # [0] is argmax

        if hmm.n > 1:
            y[m - hmm.n + 2: m + 1] = max([(pi[m][prev] + log2(hmm.trans_prob('</s>', prev)), prev) for prev in pi[m] if hmm.trans_prob('</s>', prev) > 0.0])[1]
            for k in reversed(list(range(m - hmm.n + 2))):
                y[k] = bp[k + hmm.n - 1][tuple(y[k + 1: k + hmm.n])]
            return y[1:]
        else:
            import operator
            y = list(range(len(pi)))
            for ix in range(1, len(pi) - 1):
                y[ix] = max(pi[ix].items(), key=operator.itemgetter(1))[0][0]
            return y[1:][:-1]
