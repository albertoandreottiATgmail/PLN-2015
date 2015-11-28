#!/bin/bash
#exercise 1
python tagging/scripts/stats.py

#exercise 2
nosetests tagging/tests/test_baseline.py
python tagging/scripts/train.py -o baseline
python tagging/scripts/eval.py -i baseline

#exercise 3
python tagging/scripts/train.py -o baseline
python tagging/scripts/eval.py -i baseline

#exercise 4
nosetests tagging/tests/test_hmm.py 
nosetests tagging/tests/test_viterbi_tagger.py 

#exercise 5
for i in `seq 1 4`;
    do
        python tagging/scripts/train.py -o hmm$i -m hmm:$i
        python tagging/scripts/eval.py -i hmm$i
    done  

#exercise 7
for i in `seq 1 4`;
    do
        python tagging/scripts/train.py -o memm$i -m memm:$i
        python tagging/scripts/eval.py -i memm$i
    done  