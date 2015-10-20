from features import History

class MEMM:
 
    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.sents = tagged_sents

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            [words, tags] = zip(*sent)
            print(list(zip(*sent)))
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
 
    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
 
    def tag(self, sent):
        """Tag a sentence.
 
        sent -- the sentence.
        """
 
    def tag_history(self, h):
        """Tag a history.
 
        h -- the history.
        """
 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """