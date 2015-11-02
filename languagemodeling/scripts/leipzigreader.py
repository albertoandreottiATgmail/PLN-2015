# -*- coding: latin-1 -*-
# A reader for files in the Leipzig corpus ##
# file : http://corpora2.informatik.uni-leipzig.de/downloads/eng-za_web_2013_100K-text.tar.gz ##
import re
import itertools
from nltk.corpus import PlaintextCorpusReader


class LeipzigCorpusReader(PlaintextCorpusReader):

    def __init__(self, fname):

        with open(fname) as f:
            content = f.readlines()

        self._sentences = sents = [re.sub(r'^.+\\t', '', sent) for sent in content]
        # remove first token(line number)
        self._sentences = [sentence.split()[1:] for sentence in self._sentences]
        # split period in last word
        self._sentences = [sent[:-1] + [sent[-1][:-1], '.'] for sent in sents]

        # handle the comma
        def split_comma(sent):
            answer = []
            for word in sent:
                if word.endswith(','):
                    answer.append(word[:-1])
                    answer.append(',')
                else:
                    answer.append(word)
            return answer
        self._sentences = [split_comma(sentence) for sentence in self._sentences]

        def split_hyphen(sent):
            answer = []
            for word in sent:
                for subw in word.split('-'):
                    answer.append(subw)
            return answer

        # handle the parentheses
        def split_char(sent, schar, echar):
            answer = []
            for word in sent:
                if word.endswith(echar):
                    answer.append(word[:-1])
                    answer.append(word[-1])
                elif word.startswith(schar):
                    answer.append(word[0])
                    answer.append(word)
                else:
                    answer.append(word)
            return answer

        self._sentences = [split_char(sent, '(', ')') for sent in sents]
        self._sentences = [split_char(sent, '“', '”') for sent in sents]
        self._sentences = [split_char(sent, '"', '"') for sent in sents]
        self._sentences = [split_char(sent, '\'', '\'') for sent in sents]
        self._sentences = [split_char(sent, ' ', '!') for sent in sents]
        self._sentences = [split_char(sent, ' ', '?') for sent in sents]
        self._sentences = [split_char(sent, ' ', '?') for sent in sents]
        self._sentences = [split_hyphen(sent) for sent in sents]

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
        :return: the given file(s) as a list of sentences or utterances,
        each encoded as a list of word strings.
        :rtype: list(list(str)) """

        return self._sentences
