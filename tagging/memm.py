from features import *
from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class MEMM:
 
    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.vocab = set()
        self.n = n
        self.sents = tagged_sents
        self.features = features = [next_word_lower, prev_tags, word_lower, word_isupper, word_istitle, word_isdigit, word_isfirst, word_ends_mente, word_looks_like_verb]
        self.vect = vect = Vectorizer(features)
        vect.fit(self.sents_histories(tagged_sents))
        self.model = MultinomialNB()
        
        self.tagset = set()

        for sent in tagged_sents:
            if len(sent) == 0:
                continue
            [words, tags] = zip(*sent)
            for i in range(len(words)):
                self.tagset.add(tags[i])
                self.vocab.add(words[i])
        
        self.taglist = taglist = list(self.tagset)
        self.tagidx = tagidx = {taglist[key]: key for key in range(len(self.tagset))}  

        y = np.array([tagidx[tag] for tag in list(self.sents_tags(tagged_sents))])
        #print(y)
        self.model.fit(vect.transform(self.sents_histories(tagged_sents)), y)



    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """

        for sent in tagged_sents:
            if len(sent) == 0:
                continue
            [words, tags] = zip(*sent)
            tags = tuple((self.n - 1) * ['<s>']) + tags
            for i in range(len(words)):
                yield History(list(words), tags[i : i + self.n - 1], i)


 
    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        [words, tags] = zip(*tagged_sent)
        tags = tuple((self.n - 1) * ['<s>']) + tags
        for i in range(len(words)):
            yield History(list(words), tags[i : i + self.n - 1], i)

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            if len(sent) == 0:
                continue
            [words, tags] = zip(*sent)
            for i in range(len(words)):
                yield tags[i]
 
    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        [words, tags] = zip(*tagged_sent)
        for i in range(len(words)):
            yield tags[i]
 
 
    def tag(self, sent):
        """Tag a sentence.
 
        sent -- the sentence.
        """
        next = History(sent, tuple((self.n - 1) * ['<s>']), 0)
        ans = []
        for i in range(len(sent)):
            
            ans.append(self.tag_history(next))
            next = History(sent, ans[-self.n:], i + 1)
        return ans
        
 
    def tag_history(self, h):
        """Tag a history.
 
        h -- the history.
        """
        #print(h) #, self.vect.transform([h]))
        return self.taglist[self.model.predict(self.vect.transform([h]))]
 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        return w not in self.vocab