"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict
from corpus.ancora import SimpleAncoraCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    sents = list(corpus.tagged_sents())
    counts = defaultdict(int)
    tags = defaultdict(int)
    word_num = 0
    tag_tagged = defaultdict(lambda : defaultdict(int))
    word_tags = defaultdict(set)

    for sent in sents:
        for word_tag in sent:
            counts[word_tag[0]] += 1
            tags[word_tag[1]] += 1
            tag_tagged[word_tag[1]][word_tag[0]] += 1
            word_tags[word_tag[0]].add(word_tag[1]) 
            word_num += 1
    most_frequent_tags = [tag for tag in list(reversed(sorted(tags.items(), key=lambda x : x[1])))[:10]]

    # compute the statistics
    print('sents count: {}'.format(len(sents)))
    print('word vocabulary size: {}'.format(len(counts)))
    print('tag vocabulary size: {}'.format(len(tags)))
    print('most frequent tags: {}'.format(most_frequent_tags))
    print('% of total for most frequent tags: {}'.format([x + (x[1] / float(word_num) * 100, ) for x in most_frequent_tags]))
    
    for tag_count in most_frequent_tags:
    	print(tag_count[0], list(reversed(sorted(tag_tagged[tag_count[0]].items(), key = lambda x: x[1])))[:5])
        
    for level in range(2,10):
        print('level: ', level)
        words_in_level = [k for k in word_tags if len(word_tags[k]) == level]
        nl = len(words_in_level)
        print('# of words: ', nl, nl / float(len(counts)))
        words_in_level_count = list(reversed(sorted([(w, counts[w]) for w in words_in_level], key = lambda x: x[1])))[:5]
        print ('top words in level', words_in_level_count)



    

