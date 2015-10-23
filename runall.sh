#!/bin/bash
nosetests tagging/tests/test_baseline.py
python tagging/scripts/train.py -o baseline
python tagging/scripts/eval.py -i baseline


for i in `seq 1 4`;
    do
        python tagging/scripts/train.py -o memm$i -m memm:$i
        python tagging/scripts/eval.py -i memm$i
    done  