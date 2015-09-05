## A reader for files in the Leipzig corpus ##
## file : http://corpora2.informatik.uni-leipzig.de/downloads/eng-za_web_2013_100K-text.tar.gz ##
import re
import itertools
from nltk.corpus import PlaintextCorpusReader
import os 

class LeipzigCorpusReader(PlaintextCorpusReader):
    
    def __init__(self, fname):
        with open(os.path.dirname(os.path.realpath(__file__)) + '/' + fname) as f:
            content = f.readlines()

        self._sentences = [re.sub(r'^.+\\t', '', sentence) for sentence in content]
          

    def words(self, fileids=None): 
        """ 
        :return: the given file(s) as a list of words and punctuation symbols. 
        :rtype: list(str) 
        """

        def flatten(listOfLists):
            "Flatten one level of nesting"
            return itertools.chain.from_iterable(listOfLists)

        return flatten([sentence.split() for sentence in self._sentences]) 

    def sents(self, fileids=None): 
        """ 
        :return: the given file(s) as a list of sentences or utterances, each encoded as a list of word strings. 
        :rtype: list(list(str)) """ 
    
        return [sentence.split() for sentence in self._sentences] 