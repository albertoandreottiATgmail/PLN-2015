from featureforge.feature import Feature

from collections import namedtuple

from featureforge.feature import Feature


# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')



def prev_tags(h):
    """Feature: current lowercased word.
    
    h -- a history.
    """
    sent, i = h.sent, h.i
    return h.prev_tags


def word_lower(h):
    """Feature: current lowercased word.
    
    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].lower()

def word_isupper(h):
    """Feature: is current word uppercase?
    
    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].isupper()    

def word_istitle(h):
    """Feature: is current word a title?
    
    h -- a history.
    """   
    sent, i = h.sent, h.i
    return sent[i][0].isupper() and sent[i][1:].islower()  

def word_isdigit(h):
    """Feature: current word is a digit.
    
    h -- a history.
    """   
    sent, i = h.sent, h.i
    return sent[i].isnumeric()


class NPrevTags(Feature):
 
    def __init__(self, n):
        """Feature: n previous tags tuple.
 
        n -- number of previous tags to consider.
        """
        self.n = n
 
    def _evaluate(self, h):
        """n previous tags tuple.
 
        h -- a history.
        """
        return h.prev_tags[-self.n:]
 
class PrevWord(Feature):
 
    def __init__(self, f):
        """Feature: the feature f applied to the previous word.
 
        f -- the feature.
        """
        self.f = f
 
    def _evaluate(self, h):
        """Apply the feature to the previous word in the history.
 
        h -- the history.
        """
        history = History(h.sent, h.prev_tags, h.i - 1)
        return self.f(history)

