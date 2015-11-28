from featureforge.feature import Feature
from collections import namedtuple
# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')


def prev_tags(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    return h.prev_tags


def word_lower(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].lower()


def next_word_lower(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    sent, i = h.sent, h.i
    if i < len(sent) - 1:
        return sent[i + 1].lower()
    else:
        return '</s>'


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
    return sent[i][0].isupper() and sent[i][1:].islower() and i != 0


def word_isfirst(h):
    """Feature: is current word a title?

    h -- a history.
    """
    sent, i = h.sent, h.i
    return i == 0


def word_isdigit(h):
    """Feature: current word is a digit.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].isnumeric()


def word_comes_after_verb(h):
    """Feature: current word is a digit.

    h -- a history.
    """
    return 'V' in h.prev_tags


def word_ends_mente(h):
    """Feature: current word is a digit.

    h -- a history.
    """
    return h.sent[h.i].endswith('mente')


def word_looks_like_verb(h):
    """Feature: current word ends in 'ando', 'ado', 'ar', 'er', 'ir'.

    h -- a history.
    """
    return sum([h.sent[h.i].endswith(suffix) for suffix in ['ando', 'ado', 'ar', 'er', 'ir']]) == 1


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
