#!/bin/bash
nosetests tagging/tests/test_baseline.py
python tagging/scripts/train.py -o baseline
python tagging/scripts/eval.py -i baseline
