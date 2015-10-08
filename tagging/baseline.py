
from collections import defaultdict
import operator



class BaselineTagger:

    # this should be a lambda, but lambdas cannot be pickled!
    def dd(self):
        return defaultdict(int)

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        self.ml_tag = defaultdict(self.dd)
        self.tags = defaultdict(int)
        self.counts = defaultdict(int)
        for sent in tagged_sents:
            for word_tag in sent:
                self.ml_tag[word_tag[0]][word_tag[1]] += 1
                self.tags[word_tag[1]] += 1
                self.counts[word_tag[0]] += 1

        for key in self.ml_tag:
            self.ml_tag[key] = max(self.ml_tag[key])         

        self.most_likely = max(list(self.tags.items()), key=operator.itemgetter(1))[0]

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self.most_likely
        else:
            return self.ml_tag[w]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.counts
